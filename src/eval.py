from src.utils import compute_gradient_norm, get_params_from_yaml_file, seed_everything, get_device
from src.models import get_model
from src.models.utils import load_model_checkpoint, compute_num_trainable_params
from src.models.classifiers.multihead_attentive import MultiHeadAttentiveClassifier
from src.models.classifiers.simple_attentive import SimpleAttentiveClassifier
from src.models.classifiers.cross_attentive import CrossAttentiveClassifier

from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.datasets import get_dataset
from src.metrics import TopKAccuracy, LossMetric, MeanPerClassAccuracy
from src.optimizer import get_optimizer
from src.logger import Logger

from pathlib import Path
import torch
import torch.nn.functional as F 
from tqdm import tqdm
import copy
import numpy as np

def evaluate(params):
    device = get_device(device_number=params['training']['device_number'])    
    seed_everything(seed=params['training']['seed'])

    chkpt_path = Path(params['checkpoint_path'])
    chkpt_config = chkpt_path/'config.yaml'
    chkpt_params = get_params_from_yaml_file(chkpt_config)
    print(f'Loaded CHECKPOINT params from config file: {chkpt_config}')
    dataset_name = params['dataset_name']

    # -- LOGGER --
    run_name = params['checkpoint_path'].split('/')[-1]
    logger = Logger(root_folder=params['logging']['root_folder'], 
                    run_name=run_name,
                    training_type=f'eval',
                    dataset_name=dataset_name,
                    log_to_wandb=params['logging']['log_to_wandb'],
                    config=params)

    tfms_train = get_train_transforms(crop_size=chkpt_params['model']['img_size'],
                                      crop_scale=chkpt_params['transforms']['crop_scale'],
                                      tfms_type=chkpt_params['transforms']['type'],
                                      num_views=1)
    
    tfms_test = get_test_transforms(crop_size=chkpt_params['model']['img_size'])

    dl_train, dl_test, dl_valid = get_dataset(name=dataset_name,
                                              datasets_root=chkpt_params['datasets_root'],
                                              batch_size=chkpt_params['training']['batch_size'],
                                              num_workers=chkpt_params['training']['num_workers'],
                                              tfms_train=tfms_train,
                                              tfms_test=tfms_test,
                                              seed=chkpt_params['training']['seed'])

    if dl_valid is None: 
        dl_valid = dl_test

    num_classes = len(dl_train.dataset.classes)

    # -- FROZEN MODEL --
    backbone = get_model(name=chkpt_params['model']['name'],
                         img_size=chkpt_params['model']['img_size'],
                         patch_size=chkpt_params['model']['patch_size'],
                         in_channels=chkpt_params['model']['in_channels'],
                         embed_dim=chkpt_params['model']['embed_dim'],
                         depth=chkpt_params['model']['depth'],
                         num_heads=chkpt_params['model']['num_heads'],
                         mlp_ratio=chkpt_params['model']['mlp_ratio'],
                         dropout_rate=chkpt_params['model']['dropout_rate'],
                         attention_dropout=chkpt_params['model']['attention_dropout'],
                         stochastic_depth_rate=chkpt_params['model']['stochastic_depth_rate'],
                         num_register_tokens=chkpt_params['model']['num_register_tokens'],
                         ffn_layer=chkpt_params['model']['ffn_layer'],
                         verbose=True)

    backbone = load_model_checkpoint(file_path=chkpt_path/params['checkpoint_weights'], model=backbone, device=device)
    backbone.to(device)
    backbone = backbone.eval()

    # -- CLASSIFIERS --
    multi_cls = MultiHeadAttentiveClassifier(embed_dim=chkpt_params['model']['embed_dim'],
                                             num_classes=num_classes,
                                             num_heads=4).to(device)
    
    simple_cls = SimpleAttentiveClassifier(embed_dim=chkpt_params['model']['embed_dim'],
                                           num_classes=num_classes).to(device)
    
    cross_cls = CrossAttentiveClassifier(embed_dim=chkpt_params['model']['embed_dim'],
                                         num_classes=num_classes,
                                         num_heads=4,
                                         attention_dropout=0.2,
                                         projection_dropout=0.2).to(device)
    

    logger.log_metadata(metadata_dict={
        'multi_num_trainable_params': compute_num_trainable_params(model=multi_cls),
        'cross_num_trainable_params': compute_num_trainable_params(model=cross_cls),
        'simple_num_trainable_params': compute_num_trainable_params(model=simple_cls)
    })

    # -- OPTIMIZERS -- 
    num_epochs = params['training']['num_epochs']
    ipe = len(dl_train) # iterations per epoch
    total_steps = int(num_epochs * ipe)

    get_opt = lambda model: get_optimizer(model=model,
                                          total_steps=total_steps,
                                          lr_warmup_steps=int(params['training']['lr_warmup_epochs'] * ipe),
                                          lr_start=params['training']['lr_start'],
                                          lr_peak=params['training']['lr_peak'],
                                          lr_final=params['training']['lr_final'],
                                          lr_flat_pctg=params['training']['lr_flat_pctg'],
                                          wd_start=params['training']['wd_start'],
                                          wd_final=params['training']['wd_final'])  
    multi_opt = get_opt(multi_cls)
    cross_opt = get_opt(cross_cls)
    simple_opt = get_opt(simple_cls)

    # -- TRAIN --
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 1
    best_top1_acc = 0
    best_model = None
    best_topidx = None

    for epoch_idx in range(start_epoch, num_epochs + 1):
        logger.increment_epoch()

        # Train
        multi_cls.train()
        cross_cls.train()
        simple_cls.train()

        metrics = {
            'multi': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            },
            'cross': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            },
            'simple': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            }
        }

        for batch in tqdm(dl_train, desc=f'Epoch {epoch_idx}/{num_epochs}: Train'):
            logger.increment_iteration()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = backbone(x)
                feats = F.layer_norm(feats, (feats.size(-1),)).detach()            
            
            # 1) Multi
            multi_opt.zero_grad()
            logits = multi_cls(feats)
            loss = criterion(logits, y)
            loss.backward()
            gn = compute_gradient_norm(multi_cls.parameters())
            multi_opt.step()

            metrics['multi']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['multi']['top1'].update_metric(logits=logits, targets=y)
            metrics['multi']['top5'].update_metric(logits=logits, targets=y)
            metrics['multi']['mpc'].update_metric(logits=logits, targets=y)
            logger.update_iteration(key='loss_multi', value=loss.item())
            logger.update_iteration(key='grad_norm_multi', value=gn)

            # 2) Cross
            cross_opt.zero_grad()
            logits = cross_cls(feats)
            loss = criterion(logits, y)
            loss.backward()
            gn = compute_gradient_norm(cross_cls.parameters())
            cross_opt.step()

            metrics['cross']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['cross']['top1'].update_metric(logits=logits, targets=y)
            metrics['cross']['top5'].update_metric(logits=logits, targets=y)
            metrics['cross']['mpc'].update_metric(logits=logits, targets=y)
            logger.update_iteration(key='loss_cross', value=loss.item())
            logger.update_iteration(key='grad_norm_cross', value=gn)

            # 3) Simple
            simple_opt.zero_grad()
            logits = simple_cls(feats)
            loss = criterion(logits, y)
            loss.backward()
            gn = compute_gradient_norm(simple_cls.parameters())
            simple_opt.step()

            metrics['simple']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['simple']['top1'].update_metric(logits=logits, targets=y)
            metrics['simple']['top5'].update_metric(logits=logits, targets=y)
            metrics['simple']['mpc'].update_metric(logits=logits, targets=y)
            logger.update_iteration(key='loss_cross', value=loss.item())
            logger.update_iteration(key='grad_norm_simple', value=gn)

            opt_info = simple_opt.info()
            logger.update_iteration(key='lr', value=opt_info['lr'])
            logger.update_iteration(key='wd', value=opt_info['wd'])
            logger.log_iteration()

        logger.update_epoch(key='Train Loss Multi',  value=metrics['multi']['loss'].get_value())
        logger.update_epoch(key='Train Top-1 Multi', value=metrics['multi']['top1'].get_value())
        logger.update_epoch(key='Train Top-5 Multi', value=metrics['multi']['top5'].get_value())
        logger.update_epoch(key='Train MPC Multi',   value=metrics['multi']['mpc'].get_value())

        logger.update_epoch(key='Train Loss Cross',  value=metrics['cross']['loss'].get_value())
        logger.update_epoch(key='Train Top-1 Cross', value=metrics['cross']['top1'].get_value())
        logger.update_epoch(key='Train Top-5 Cross', value=metrics['cross']['top5'].get_value())
        logger.update_epoch(key='Train MPC Cross',   value=metrics['cross']['mpc'].get_value())

        logger.update_epoch(key='Train Loss Simple',  value=metrics['simple']['loss'].get_value())
        logger.update_epoch(key='Train Top-1 Simple', value=metrics['simple']['top1'].get_value())
        logger.update_epoch(key='Train Top-5 Simple', value=metrics['simple']['top5'].get_value())
        logger.update_epoch(key='Train MPC Simple',   value=metrics['simple']['mpc'].get_value())

        # Eval
        multi_cls.eval()
        cross_cls.eval()
        simple_cls.eval()

        metrics = {
            'multi': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            },
            'cross': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            },
            'simple': {
                'loss': LossMetric(),
                'top1': TopKAccuracy(k=1),
                'top5': TopKAccuracy(k=5),
                'mpc': MeanPerClassAccuracy(num_classes=num_classes)
            }
        }

        for batch in tqdm(dl_valid, desc=f'Epoch {epoch_idx}/{num_epochs}: Valid'):
            x, y = batch
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = backbone(x)
                feats = F.layer_norm(feats, (feats.size(-1),)).detach()     
            
            # 1) Multi
            logits = multi_cls(feats)
            loss = criterion(logits, y)
            metrics['multi']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['multi']['top1'].update_metric(logits=logits, targets=y)
            metrics['multi']['top5'].update_metric(logits=logits, targets=y)
            metrics['multi']['mpc'].update_metric(logits=logits, targets=y)

            # 2) Cross
            logits = cross_cls(feats)
            loss = criterion(logits, y)
            metrics['cross']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['cross']['top1'].update_metric(logits=logits, targets=y)
            metrics['cross']['top5'].update_metric(logits=logits, targets=y)
            metrics['cross']['mpc'].update_metric(logits=logits, targets=y)

            # 3) Simple
            logits = simple_cls(feats)
            loss = criterion(logits, y)
            metrics['simple']['loss'].update_metric(value=loss.item(), num_samples=len(x))
            metrics['simple']['top1'].update_metric(logits=logits, targets=y)
            metrics['simple']['top5'].update_metric(logits=logits, targets=y)
            metrics['simple']['mpc'].update_metric(logits=logits, targets=y)

        logger.update_epoch(key='Valid Loss Multi',  value=metrics['multi']['loss'].get_value())
        logger.update_epoch(key='Valid Top-1 Multi', value=metrics['multi']['top1'].get_value())
        logger.update_epoch(key='Valid Top-5 Multi', value=metrics['multi']['top5'].get_value())
        logger.update_epoch(key='Valid MPC Multi',   value=metrics['multi']['mpc'].get_value())

        logger.update_epoch(key='Valid Loss Cross',  value=metrics['cross']['loss'].get_value())
        logger.update_epoch(key='Valid Top-1 Cross', value=metrics['cross']['top1'].get_value())
        logger.update_epoch(key='Valid Top-5 Cross', value=metrics['cross']['top5'].get_value())
        logger.update_epoch(key='Valid MPC Cross',   value=metrics['cross']['mpc'].get_value())

        logger.update_epoch(key='Valid Loss Simple',  value=metrics['simple']['loss'].get_value())
        logger.update_epoch(key='Valid Top-1 Simple', value=metrics['simple']['top1'].get_value())
        logger.update_epoch(key='Valid Top-5 Simple', value=metrics['simple']['top5'].get_value())
        logger.update_epoch(key='Valid MPC Simple',   value=metrics['simple']['mpc'].get_value())

        top1 = [metrics['multi']['top1'].get_value(), 
                metrics['cross']['top1'].get_value(), 
                metrics['simple']['top1'].get_value()]
        top1_idx = np.argmax(top1)
        
        if top1[top1_idx] >= best_top1_acc:
            best_top1_acc = top1[top1_idx]
            best_model = copy.deepcopy([multi_cls, cross_cls, simple_cls][top1_idx])
            best_topidx = top1_idx

        logger.log_epoch()
        
    best_model.eval()
    top1_acc = TopKAccuracy(k=1)
    top5_acc = TopKAccuracy(k=5)
    mpc_acc = MeanPerClassAccuracy(num_classes=num_classes)

    for batch in tqdm(dl_test, desc=f'Best model test'):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            feats = backbone(x)
            feats = F.layer_norm(feats, (feats.size(-1),)).detach()   
            logits = best_model(feats)
        top1_acc.update_metric(logits=logits, targets=y)
        top5_acc.update_metric(logits=logits, targets=y)
        mpc_acc.update_metric(logits=logits, targets=y)

    logger.log_test(top1_accuracy=top1_acc.get_value(), top5_accuracy=top5_acc.get_value(), mpc_accuracy=mpc_acc.get_value())

    print(f'Best model: {["multi","cross","simple"][best_topidx]}')
    print(f'Top-1 Accuracy: {top1_acc.get_value()}')
    print(f'Top-5 Accuracy: {top5_acc.get_value()}')
    print(f'Mean-Per-Class Accuracy: {mpc_acc.get_value()}')
    print(mpc_acc.get_top1_per_class())