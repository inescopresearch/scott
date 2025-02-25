from src.models.scott import SCOTT
from src.models.vit import ViT
from src.models.utils import compute_num_trainable_params

MODELS = {
    'scott': SCOTT,
    'vit': ViT
}

MODELS2NAMES = {
    'scott': 'Sparse Convolutional Tokenizer for Transformer (SCOTT)',
    'vit': 'Vision Transformer (ViT)'
}

def get_model(name,
              img_size,
              patch_size,
              embed_dim,
              in_channels,
              depth,
              num_heads,
              mlp_ratio,
              dropout_rate,
              attention_dropout,
              stochastic_depth_rate,
              num_register_tokens,
              ffn_layer,
              verbose=True):
    assert name in MODELS.keys(), f'Error invalid model name. Choose one of: {MODELS.keys()}'

    model =  MODELS[name](img_size=img_size,
                          patch_size=patch_size,
                          embed_dim=embed_dim,
                          in_channels=in_channels,
                          depth=depth,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          dropout_rate=dropout_rate,
                          attention_dropout=attention_dropout,
                          stochastic_depth_rate=stochastic_depth_rate,
                          num_register_tokens=num_register_tokens,
                          ffn_layer=ffn_layer)
    
    num_trainable_params = compute_num_trainable_params(model)

    if verbose:
        print(f'Creating model: {MODELS2NAMES[name]}.')
        print(f'- Depth: {depth}.')
        print(f'- Embed dim: {embed_dim}.')
        print(f'- Num heads: {num_heads}.')
        print(f'- FFN layer: {ffn_layer}.')
        print(f'- Num patches: {model.num_patches}')
        print(f'- Num register tokens: {model.num_register_tokens}.')

        print(f'- Num trainable params: {num_trainable_params}.')

    return model

