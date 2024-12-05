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

from base_miner.DFB.optimizor.SAM import SAM
from base_miner.DFB.optimizor.LinearLR import LinearDecayLR
from base_miner.DFB.config.helpers import save_config
from base_miner.DFB.trainer.trainer import Trainer
from base_miner.DFB.detectors import DETECTOR
from base_miner.DFB.metrics.utils import parse_metric_for_print
from base_miner.DFB.logger import create_logger, RankFilter

from huggingface_hub import hf_hub_download

# BitMind imports (not from original Deepfake Bench repo)
from base_miner.datasets.util import load_and_split_datasets, create_real_fake_datasets
from base_miner.config import VIDEO_DATASETS, IMAGE_DATASETS, FACE_IMAGE_DATASETS
from bitmind.utils.image_transforms import (
    get_base_transforms, 
    get_random_augmentations, 
    get_ucf_base_transforms, 
    get_tall_base_transforms
)
from base_miner.DFB.config.constants import (
    CONFIG_PATHS,
    WEIGHTS_DIR,
    HF_REPOS
)

TRANSFORM_FNS = {
    'UCF': get_ucf_base_transforms,
    'TALL': get_tall_base_transforms
}


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector', type=str, choices=['UCF', 'TALL'], required=True, help='Detector name')
parser.add_argument('--modality', type=str, default='image', choices=['image', 'video'])
parser.add_argument('--faces_only', dest='faces_only', action='store_true', default=False)
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda',
                    help='Specify whether to use CPU or GPU. Defaults to GPU if available.')
parser.add_argument('--gpu_id', type=int, default=0, help='Specify the GPU ID to use if using GPU. Defaults to 0.')
parser.add_argument('--workers', type=int, default=os.cpu_count() - 1,
                    help='number of workers for data loading')
parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
args = parser.parse_args()


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_datasets(config, logger):
    start_time = log_start_time(logger, "Loading and splitting individual datasets")

    fake_datasets = load_and_split_datasets(
        config['dataset_meta']['fake'], modality=config['modality'], split_transforms=config['split_transforms'])
    real_datasets = load_and_split_datasets(
        config['dataset_meta']['real'], modality=config['modality'], split_transforms=config['split_transforms'])

    log_finish_time(logger, "Loading and splitting individual datasets", start_time)
    
    start_time = log_start_time(logger, "Creating real fake dataset splits")
    train_dataset, val_dataset, test_dataset, source_label_mapping = create_real_fake_datasets(
        real_datasets,
        fake_datasets,
        source_labels=True,  # TODO UCF Only
        group_sources_by_name=True)

    log_finish_time(logger, "Creating real fake dataset splits", start_time)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=config['workers'],
        drop_last=True,
        collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=config['workers'],
        drop_last=True,
        collate_fn=val_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['train_batchSize'],
        shuffle=True, 
        num_workers=config['workers'],
        drop_last=True,
        collate_fn=train_dataset.collate_fn)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, source_label_mapping


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
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
        scheduler = None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))
    return scheduler


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    gc.collect()

    detector_config_path = CONFIG_PATHS[args.detector]

    # parse options and load config
    with open(detector_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['log_dir'] = os.getcwd()
    config['device'] = args.device
    config['modality'] = args.modality
    config['workers'] = args.workers
    config['gpu_id'] = args.gpu_id
    if args.epochs:
        config['nEpochs'] = args.epochs

    tforms = TRANSFORM_FNS.get(args.detector, None)((256, 256))
    config['split_transforms'] = {
        'train': tforms,
        'validation': tforms,
        'test': tforms
    }

    if config['modality'] == 'video':
        config['dataset_meta'] = VIDEO_DATASETS
    elif config['modality'] == 'image':
        if args.faces_only:
            config['dataset_meta'] = FACE_IMAGE_DATASETS
        else:
            config['dataset_meta'] = IMAGE_DATASETS

    dataset_names = [item["path"] for datasets in config['dataset_meta'].values() for item in datasets]
    config['train_dataset'] = dataset_names
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat

    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outputs_dir = os.path.join(config['log_dir'], 'logs', config['model_name'] + '_' + timenow)
    config['log_dir'] = outputs_dir

    os.makedirs(outputs_dir, exist_ok=True)
    logger = create_logger(os.path.join(outputs_dir, 'training.log'))
    logger.info('Save log to {}'.format(outputs_dir))
    
    config['ddp']= args.ddp

    # prepare the data loaders
    train_loader, val_loader, test_loader, source_label_mapping = prepare_datasets(config, logger)
    config['specific_task_number'] = len(set(source_label_mapping.values()))

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

    # download weights if huggingface repo provided.
    # Note: TALL currently skips this and downloads from github
    pretrained_config = config.get('pretrained', {})
    if not isinstance(pretrained_config, str):
        hf_repo = pretrained_config.get('hf_repo')
        weights_filename = pretrained_config.get('filename')
        if hf_repo and weights_filename:
            local_path = Path(WEIGHTS_DIR) / weights_filename
            if not local_path.exists():
                model_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=weights_filename,
                    local_dir=WEIGHTS_DIR
                )
                logger.info(f"Downloaded {hf_repo}/{weights_filename} to {model_path}")
            else:
                model_path = local_path
                logger.info(f"{model_path} exists, skipping download")
            config['pretrained']['local_path'] = str(model_path)
    else:
        logger.info("Pretrain config is a url, falling back to detector-specific download")

    # prepare model and trainer
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(config['device'])

    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    trainer = Trainer(config, model, config['device'], optimizer, scheduler, logger, metric_scoring)

    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # save training configs
    save_config(config, outputs_dir)

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

    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
