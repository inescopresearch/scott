from pathlib import Path
import os
import yaml
import time
import wandb
import csv

class Logger():

    def __init__(self, root_folder, training_type, dataset_name, run_name, log_to_wandb, config):
        self.root_folder = Path(root_folder)
        if not self.root_folder.exists():
            os.mkdir(self.root_folder)
        self.training_path = self.root_folder/training_type
        if not self.training_path.exists():
            os.mkdir(self.training_path)
        self.dataset_path = self.training_path/dataset_name
        if not self.dataset_path.exists():
            os.mkdir(self.dataset_path)
        self.run_name = run_name
        self.log_path = self.dataset_path/self.run_name
        if not self.log_path.exists():
            os.mkdir(self.log_path)
        else:
            raise Exception(f'Error: path {self.log_path} already exists.')
        
        self.project_name = f'{training_type}_{dataset_name}'
        self.iterations_fpath = self.log_path/'iterations.csv'
        self.epochs_fpath = self.log_path/'epochs.csv'
        self.config_fpath = self.log_path/'config.yaml'
        self.test_fpath = self.log_path/'test.csv'
        self.metadata_fpath = self.log_path/'metadata.csv'
        self.log_to_wandb=log_to_wandb
        if self.log_to_wandb:
            wandb.login()
            wandb.init(project=self.project_name, name=f"{self.run_name}", config=config)

        self.data_iteration = dict()
        self.data_epoch = dict()
        self.epoch_step = 0
        self.iter_step = 0

        self.save_config_file(config)
    
    @staticmethod
    def get_run_name(model_name, depth, patch_size, img_size):
        return f'{model_name}_{int(depth)}_{int(patch_size)}_{int(img_size)}_{str(time.time()).split(".")[0]}'
    
    def save_config_file(self, args):
        with open(self.config_fpath, 'w') as f:
            yaml.dump(args, f)

    def update_iteration(self, key, value):
        self.data_iteration[key] = value

    def update_epoch(self, key, value):
        self.data_epoch[key] = value

    def increment_iteration(self):
        self.iter_step += 1

    def increment_epoch(self):
        self.epoch_step += 1

    def _log_to_file(self, fpath, data):
        write_header = not fpath.exists()
        with open(fpath, 'a', newline='') as f:
            csvwriter = csv.writer(f)
            if write_header:
                csvwriter.writerow(list(data.keys())) 
            csvwriter.writerow(data.values())

    def log_iteration(self):
        # Log to wandb
        if self.iter_step == 1 and self.log_to_wandb:
            for key in self.data_iteration.keys():
                wandb.define_metric(key, step_metric='iter_step')
        
        self.data_iteration['iter_step'] = self.iter_step 
        if self.log_to_wandb:
            wandb.log(self.data_iteration)
        
        # Log to file
        self._log_to_file(fpath=self.iterations_fpath, data=self.data_iteration)
            
        # Reset
        self.data_iteration = dict()

    def log_epoch(self):
        # Log to wandb
        if self.epoch_step == 1 and self.log_to_wandb:
            for key in self.data_epoch.keys():
                wandb.define_metric(key, step_metric='epoch_step')
        
        self.data_epoch['epoch_step'] = self.epoch_step 
        if self.log_to_wandb:
            wandb.log(self.data_epoch)
        
        # Log to file
        self._log_to_file(fpath=self.epochs_fpath, data=self.data_epoch)

        # Reset
        self.data_epoch = dict()
    
    def log_test(self, top1_accuracy=None, top5_accuracy=None, mpc_accuracy=None):
        data = dict()
        if top1_accuracy:
            data['Test Top-1'] = top1_accuracy
        if top5_accuracy:    
            data['Test Top-5'] = top5_accuracy
        if mpc_accuracy:
            data['Test Mean-Per-Class Top-1'] = mpc_accuracy 
            
        if self.log_to_wandb:
            wandb.log(data)
        # Log to file
        self._log_to_file(fpath=self.test_fpath, data=data)

    def log_metadata(self, metadata_dict):
        if self.log_to_wandb:
            wandb.log(metadata_dict)
        # Log to file
        self._log_to_file(fpath=self.metadata_fpath, data=metadata_dict)

    def save_config_file(self, args):
        with open(self.config_fpath, 'w') as f:
            yaml.dump(args, f)