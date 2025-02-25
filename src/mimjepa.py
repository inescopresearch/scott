from src.utils import seed_everything, get_device, compute_gradient_norm
from src.models.utils import compute_num_trainable_params, save_model_checkpoint
from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.datasets import get_dataset
from src.logger import Logger
from src.masks.mask_generator import MaskGenerator
from src.models import get_model
from src.models.predictor import Predictor
from src.models.classifiers.simple_attentive import SimpleAttentiveClassifier as Classifier
from src.optimizer import get_optimizer
from src.schedulers import LinearScheduler, CosineScheduler
from src.metrics import TopKAccuracy, LossMetric, MeanPerClassAccuracy

import torch
from tqdm import tqdm
import copy 
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, s_preds, s_targets, keep_masks):
        B, N, D = s_preds.shape
        drop_masks = (1 - keep_masks).detach()
        s_preds = s_preds * drop_masks.unsqueeze(-1)
        s_targets = s_targets * drop_masks.unsqueeze(-1)
        recon_loss = self.criterion(s_preds, s_targets)
        recon_loss = ((recon_loss.sum(-1) + 1e-8)/ D).sum(-1) / (drop_masks.sum(-1) + 1e-8)
        recon_loss = recon_loss.mean()
        return recon_loss
    
def mimjepa_training(params):
    # -- SETUP --
    device = get_device(device_number=params['training']['device_number'])    
    seed_everything(seed=params['training']['seed'])
    
    # -- LOGGER --
    run_name = Logger.get_run_name(model_name=params['model']['name'],
                                   depth=params['model']['depth'],
                                   patch_size=params['model']['patch_size'],
                                   img_size=params['model']['img_size'])
                                   
    logger = Logger(root_folder=params['logging']['root_folder'], 
                    run_name=run_name,
                    training_type=params['training_type'],
                    dataset_name=params["dataset_name"],
                    log_to_wandb=params['logging']['log_to_wandb'],
                    config=params)

    # -- DATA --
    tfms_train = get_train_transforms(crop_size=params['model']['img_size'],
                                      crop_scale=params['transforms']['crop_scale'],
                                      tfms_type=params['transforms']['type'],
                                      num_views=2)
        
    tfms_test = get_test_transforms(crop_size=params['model']['img_size'])

    dl_train, dl_test, _ = get_dataset(name=params['dataset_name'],
                                       datasets_root=params['datasets_root'],
                                       batch_size=params['training']['batch_size'],
                                       num_workers=params['training']['num_workers'],
                                       tfms_train=tfms_train,
                                       tfms_test=tfms_test,
                                       seed=params['training']['seed'])
    
    num_classes = len(dl_train.dataset.classes)

    # -- MASKING --
    mask_generator = MaskGenerator(img_size=params['model']['img_size'],
                                   patch_size=params['model']['patch_size'],
                                   masking_type=params['masking']['type'])
    
    def generate_masks_batch(batch_size):
        """1 = keep patch / 0 = drop patch."""
        keep_masks = mask_generator(batch_size=batch_size,
                                    mask_ratio=params['masking']['ratio']).to(device)
        return keep_masks
    
    # -- CONTEXT ENCODER --
    context_encoder = get_model(name=params['model']['name'],
                                img_size=params['model']['img_size'],
                                patch_size=params['model']['patch_size'],
                                in_channels=params['model']['in_channels'],
                                embed_dim=params['model']['embed_dim'],
                                depth=params['model']['depth'],
                                num_heads=params['model']['num_heads'],
                                mlp_ratio=params['model']['mlp_ratio'],
                                dropout_rate=params['model']['dropout_rate'],
                                attention_dropout=params['model']['attention_dropout'],
                                stochastic_depth_rate=params['model']['stochastic_depth_rate'],
                                num_register_tokens=params['model']['num_register_tokens'],
                                ffn_layer=params['model']['ffn_layer'],
                                verbose=True).to(device)
    
    logger.log_metadata(metadata_dict={
        'num_trainable_params': compute_num_trainable_params(model=context_encoder)
    })

    # -- PREDICTOR --
    predictor =  Predictor(embed_dim=context_encoder.embed_dim,
                           num_patches=context_encoder.num_patches,
                           num_register_tokens=context_encoder.num_register_tokens,
                           backbone_depth=context_encoder.depth,
                           depth=params['predictor']['depth'],
                           num_heads=context_encoder.num_heads,
                           mlp_ratio=context_encoder.mlp_ratio,
                           dropout_rate=context_encoder.dropout_rate,
                           attention_dropout=context_encoder.attention_dropout,
                           stochastic_depth_rate=context_encoder.stochastic_depth_rate,
                           ffn_layer=context_encoder.ffn_layer).to(device)
    
    # -- TARGET ENCODER --
    target_encoder = copy.deepcopy(context_encoder)

    # -- CLASSIFIER --
    classifier = Classifier(embed_dim=params['model']['embed_dim'],
                            num_classes=num_classes).to(device) 
    
    # -- MIM-JEPA SSL OPTIMIZER --
    num_epochs = params['training']['num_epochs']
    ipe = len(dl_train) # iterations per epoch
    total_steps = int(num_epochs * ipe)

    optimizer_ssl = get_optimizer(model=nn.ModuleList([context_encoder, predictor]),
                                  total_steps=total_steps,
                                  lr_warmup_steps=int(params['training']['lr_warmup_epochs'] * ipe),
                                  lr_start=params['training']['lr_start'],
                                  lr_peak=params['training']['lr_peak'],
                                  lr_final=params['training']['lr_final'],
                                  lr_flat_pctg=params['training']['lr_flat_pctg'],
                                  wd_start=params['training']['wd_start'],
                                  wd_final=params['training']['wd_final'])
    
    # -- SL OPTIMIZER -- * (Just to evaluate training progress on a downstream task).
    optimizer_sl = get_optimizer(model=classifier,
                                 total_steps=total_steps,
                                 lr_warmup_steps=int(params['training']['lr_warmup_epochs'] * ipe),
                                 lr_start=params['training']['lr_start'],
                                 lr_peak=params['training']['lr_peak'],
                                 lr_final=params['training']['lr_final'],
                                 lr_flat_pctg=params['training']['lr_flat_pctg'],
                                 wd_start=params['training']['wd_start'],
                                 wd_final=params['training']['wd_final'])
    
    # -- TRAINING LOOP --
    start_epoch = 1
    scheduler = LinearScheduler if params['training']['ema_schedule'] == 'linear' else CosineScheduler
    ema_schedule = scheduler(start_value=params['training']['ema_start'],
                             final_value=params['training']['ema_final'],
                             total_steps=total_steps)
    
    criterion_ssl = ReconstructionLoss()
    criterion_sl = torch.nn.CrossEntropyLoss()

    for epoch_idx in range(start_epoch, num_epochs + 1):
        logger.increment_epoch()

        # Train
        context_encoder.train()
        predictor.train()
        classifier.train()
        target_encoder.eval()

        loss_metric_ssl = LossMetric()
        
        loss_metric_sl = LossMetric()
        top1_acc_target = TopKAccuracy(k=1)
        top5_acc_target = TopKAccuracy(k=5)
        
        for batch in tqdm(dl_train, desc=f'Epoch {epoch_idx}/{num_epochs}: Train'):
            logger.increment_iteration()
            (x_context, x_targets), y = batch
            x_context = x_context.to(device)
            x_targets = x_targets.to(device)
            y = y.to(device)

            # -- SELF-SUPERVISED LEARNING --
            keep_masks = generate_masks_batch(batch_size=x_context.shape[0])
            
            # 1. Forward Target
            with torch.no_grad():
                s_targets = target_encoder(x_targets, masks=None)
                s_targets = F.layer_norm(s_targets, (s_targets.size(-1),)).detach()
            
            # 2. Forward Context
            optimizer_ssl.zero_grad()
            s_context = context_encoder(x_context, keep_masks)
            s_preds = predictor(s_context)
            
            # 3. Compute loss on masked patches.
            loss_ssl = criterion_ssl(s_preds, s_targets, keep_masks)
            loss_ssl.backward()
            
            # Logging stats
            grad_norm_tokenizer = compute_gradient_norm(context_encoder.tokenizer.parameters())
            grad_norm_transformer = compute_gradient_norm(context_encoder.transformer.parameters())
            grad_norm_predictor = compute_gradient_norm(predictor.parameters())

            # 4. Update context-encoder and predictor weights.
            optimizer_ssl.step()

            # 5. Exponential Moving Average of target-encoder.
            ema_value = ema_schedule.step()
            with torch.no_grad():
                for p_online, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
                    p_target.mul_(ema_value).add_((1. - ema_value) * p_online.detach().data)

            # Log training progress.
            loss_metric_ssl.update_metric(value=loss_ssl.item(), num_samples=len(x_context))
            ssl_opt_info = optimizer_ssl.info()
            logger.update_iteration(key='ssl_lr', value=ssl_opt_info['lr'])
            logger.update_iteration(key='ssl_wd', value=ssl_opt_info['wd'])
            logger.update_iteration(key='ssl_loss', value=loss_ssl.item())
            logger.update_iteration(key='grad_norm_tokenizer', value=grad_norm_tokenizer)
            logger.update_iteration(key='grad_norm_transformer', value=grad_norm_transformer)
            logger.update_iteration(key='grad_norm_predictor', value=grad_norm_predictor)
            logger.update_iteration(key='ema_value', value=ema_value)

            # -- SUPERVISED LEARNING (just to evaluate training progress on a downstream task.) --
            optimizer_sl.zero_grad()
            logits = classifier(s_targets)
            loss_sl = criterion_sl(logits, y)
            loss_sl.backward()

            grad_norm_classifier = compute_gradient_norm(classifier.parameters())
            optimizer_sl.step()

            top1_acc_target.update_metric(logits=logits, targets=y)
            top5_acc_target.update_metric(logits=logits, targets=y)
            loss_metric_sl.update_metric(value=loss_sl.item(), num_samples=len(x_context))

            sl_opt_info = optimizer_sl.info()
            logger.update_iteration(key='sl_lr', value=sl_opt_info['lr'])
            logger.update_iteration(key='sl_wd', value=sl_opt_info['wd'])
            logger.update_iteration(key='sl_loss', value=loss_sl.item())
            logger.update_iteration(key='grad_norm_classifier', value=grad_norm_classifier)

            logger.log_iteration()


        target_encoder.eval()
        classifier.eval()
        top1_acc_test = TopKAccuracy(k=1)
        top5_acc_test = TopKAccuracy(k=5)
        mpc_acc_test = MeanPerClassAccuracy(num_classes=num_classes)

        for batch in tqdm(dl_test, desc=f'Test set evaluation'):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = target_encoder(x)
                feats = F.layer_norm(feats, (feats.size(-1),)).detach()
                logits = classifier(feats)
                top1_acc_test.update_metric(logits=logits, targets=y)
                top5_acc_test.update_metric(logits=logits, targets=y)
                mpc_acc_test.update_metric(logits=logits, targets=y)

        logger.update_epoch(key='SSL Train Loss', value=loss_metric_ssl.get_value())
        logger.update_epoch(key='SL Train Loss', value=loss_metric_sl.get_value())    
        logger.update_epoch(key='Train Top-1 Target', value=top1_acc_target.get_value())
        logger.update_epoch(key='Train Top-5 Target', value=top5_acc_target.get_value())
        logger.update_epoch(key='Test Top-1', value=top1_acc_test.get_value())
        logger.update_epoch(key='Test Top-5', value=top5_acc_test.get_value())
        logger.update_epoch(key='Test MPC', value=mpc_acc_test.get_value())

        logger.log_epoch()

    _ = save_model_checkpoint(root_path=logger.log_path, model=context_encoder, tag='context-encoder')
    _ = save_model_checkpoint(root_path=logger.log_path, model=predictor, tag='predictor')
    _ = save_model_checkpoint(root_path=logger.log_path, model=target_encoder, tag='target-encoder')

    target_encoder.eval()
    classifier.eval()

    top1_acc = TopKAccuracy(k=1)
    top5_acc = TopKAccuracy(k=5)
    mpc_acc = MeanPerClassAccuracy(num_classes=num_classes)

    for batch in tqdm(dl_test, desc=f'Test set evaluation'):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            feats = target_encoder(x)
            logits = classifier(feats)
            top1_acc.update_metric(logits=logits, targets=y)
            top5_acc.update_metric(logits=logits, targets=y)
            mpc_acc.update_metric(logits=logits, targets=y)

    logger.log_test(top1_accuracy=top1_acc.get_value(), 
                    top5_accuracy=top5_acc.get_value(), 
                    mpc_accuracy=mpc_acc.get_value())

    print(f'Run completed.')
    print(f'- Test Top-1 Accuracy: {top1_acc.get_value()}')
    print(f'- Test Top-5 Accuracy: {top5_acc.get_value()}')
    print(f'- Test Mean-Per-Class Accuracy: {mpc_acc.get_value()}')
