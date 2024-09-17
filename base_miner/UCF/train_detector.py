# This script was adapted from the DeepfakeBench training code,
# originally authored by Zhiyuan Yan (zhiyuanyan@link.cuhk.edu.cn)

# Original: https://github.com/SCLBD/DeepfakeBench/blob/main/training/train.py

# BitMind's modifications include adding a testing phase, changing the 
# data load/split pipeline to work with subnet 34's image augmentations
# and datasets from BitMind HuggingFace repositories, quality of life CLI args,
# logging changes, etc.

import os
import sys
import argparse
from os.path import join
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image
from pathlib import Path
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter

from huggingface_hub import hf_hub_download

# BitMind imports (not from original Deepfake Bench repo)
from bitmind.dataset_processing.load_split_data import load_datasets, create_real_fake_datasets
from bitmind.image_transforms import base_transforms, random_aug_transforms
from bitmind.constants import DATASET_META, FACE_TRAINING_DATASET_META
from config.constants import (
    CONFIG_PATH,
    WEIGHTS_PATH,
    WEIGHTS_HF_PATH,
    BACKBONE_CKPT
)

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, default=CONFIG_PATH, help='path to detector YAML file')
parser.add_argument('--faces_only', dest='faces_only', action='store_true', default=False)
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--specific_tasks', type=int, default=None, help='specify the number of forgery methods')

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
print(f"torch.cuda.device(0): {torch.cuda.device(0)}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

def ensure_backbone_is_available(logger,
                                 weights_dir=WEIGHTS_PATH,
                                 model_filename=BACKBONE_CKPT,
                                 hugging_face_repo_name=WEIGHTS_HF_PATH):
    
    destination_path = Path(weights_dir) / Path(model_filename)
    if not destination_path.parent.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory {destination_path.parent}.")
    if not destination_path.exists():
        model_path = hf_hub_download(hugging_face_repo_name, model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        torch.save(model, destination_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Downloaded backbone {model_filename} to {destination_path}.")
    else:
        logger.info(f"{model_filename} backbone already present at {destination_path}.")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

def custom_collate_fn(batch):
    images, labels, source_labels = zip(*batch)
    
    images = torch.stack(images, dim=0)  # Stack image tensors into a single tensor
    labels = torch.LongTensor(labels) 
    source_labels = torch.LongTensor(source_labels) 
    
    data_dict = {
        'image': images,
        'label': labels,
        'label_spe': source_labels,
        'landmark': None,
        'mask': None
    }    
    return data_dict

def prepare_datasets(config, logger):
    start_time = log_start_time(logger, "Loading and splitting individual datasets")
    
    real_datasets, fake_datasets = load_datasets(dataset_meta=config['dataset_meta'], 
                                                 expert=config['faces_only'],
                                                 split_transforms=config['split_transforms'])

    log_finish_time(logger, "Loading and splitting individual datasets", start_time)
    
    start_time = log_start_time(logger, "Creating real fake dataset splits")
    train_dataset, val_dataset, test_dataset = \
    create_real_fake_datasets(real_datasets,
                              fake_datasets,
                              config['split_transforms']['train']['transform'],
                              config['split_transforms']['validation']['transform'],
                              config['split_transforms']['test']['transform'],
                              source_labels=True,
                              normalize_config={'mean': config['mean'],
                                                'std': config['std']})

    log_finish_time(logger, "Creating real fake dataset splits", start_time)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train_batchSize'],
                                               shuffle=True,
                                               num_workers=config['workers'],
                                               drop_last=True,
                                               collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['train_batchSize'],
                                             shuffle=True,
                                             num_workers=config['workers'],
                                             drop_last=True,
                                             collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config['train_batchSize'],
                                              shuffle=True, 
                                              num_workers=config['workers'],
                                              drop_last=True,
                                              collate_fn=custom_collate_fn)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader

def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))

def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def log_start_time(logger, process_name):
    """Log the start time of a process."""
    start_time = time.time()
    logger.info(f"{process_name} Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    return start_time

def log_finish_time(logger, process_name, start_time):
    """Log the finish time and elapsed time of a process."""
    finish_time = time.time()
    elapsed_time = finish_time - start_time

    # Convert elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Log the finish time and elapsed time
    logger.info(f"{process_name} Finish Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time))}")
    logger.info(f"{process_name} Elapsed Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

def main():
    torch.cuda.empty_cache()
    gc.collect()
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(os.getcwd() + '/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False

    config['split_transforms'] = {'train': {'name': 'random_aug_transforms',
                                            'transform': random_aug_transforms},
                                  'validation': {'name': 'base_transforms',
                                                 'transform': base_transforms},
                                  'test': {'name': 'base_transforms',
                                           'transform': base_transforms}}
    config['faces_only'] = args.faces_only
    config['dataset_meta'] = FACE_TRAINING_DATASET_META if config['faces_only'] else DATASET_META
    dataset_names = [item["path"] for datasets in config['dataset_meta'].values() for item in datasets]
    config['train_dataset'] = dataset_names
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    if args.specific_tasks:
        config['specific_task_number'] = args.specific_tasks
    else: config['specific_task_number'] = len(config['dataset_meta']["fake"]) + 1
    
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
        
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    outputs_dir =  os.path.join(
                config['log_dir'],
                config['model_name'] + '_' + timenow
            )
    
    os.makedirs(outputs_dir, exist_ok=True)
    logger = create_logger(os.path.join(outputs_dir, 'training.log'))
    config['log_dir'] = outputs_dir
    logger.info('Save log to {}'.format(outputs_dir))
    
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))

    ensure_backbone_is_available(logger=logger,
                                 model_filename=config['pretrained'].split('/')[-1],
                                 hugging_face_repo_name='bitmind/' + config['model_name'])
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    
    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # prepare the data loaders
    train_loader, val_loader, test_loader = prepare_datasets(config, logger)
    
    # start training
    start_time = log_start_time(logger, "Training")
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch,
                    train_data_loader=train_loader,
                    validation_data_loaders={'val':val_loader}
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with validation {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Validation metric {}".format(parse_metric_for_print(best_metric))) 
    log_finish_time(logger, "Training", start_time)
   
    # test
    start_time = log_start_time(logger, "Test")
    trainer.eval(eval_data_loaders={'test':test_loader}, eval_stage="test")
    log_finish_time(logger, "Test", start_time)
    
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()