import os
import argparse
from os.path import join
#import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
#from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter

# BitMind imports (not from original Deepfake Bench repo)
from torch.utils.data import DataLoader

from bitmind.image_transforms import base_transforms, random_aug_transforms
from util.data import load_datasets, create_real_fake_datasets

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default=os.getcwd() + '/config/ucf.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


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

def prepare_datasets(config):
    real_datasets, fake_datasets = load_datasets()
    train_dataset, val_dataset, test_dataset = create_real_fake_datasets(
        real_datasets,
        fake_datasets,
        train_transforms=random_aug_transforms,
        val_transforms=base_transforms,
        test_transforms=base_transforms,
        config=config,
        normalize=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['train_batchSize'], shuffle=True, num_workers=config['workers'], collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['train_batchSize'], shuffle=False, num_workers=config['workers'], collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['train_batchSize'], shuffle=False, num_workers=config['workers'], collate_fn=custom_collate_fn)

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
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = ""
    
    if 'task_target' in config.keys():
        task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
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
    train_loader, val_loader, test_loader = prepare_datasets(config)

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