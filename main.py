import argparse
from src.utils import get_params_from_yaml_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='name of config file to load', default='template.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    params = get_params_from_yaml_file(fpath=args.config_file)

    if params['training_type'] == 'ssl':
        from src.mimjepa import mimjepa_training
        mimjepa_training(params)
    elif params['training_type'] ==  'eval':
        from src.eval import evaluate
        evaluate(params)
    else:
        print(f'Invalid training type: { params["training_type"] }')