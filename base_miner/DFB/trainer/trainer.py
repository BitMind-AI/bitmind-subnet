# This script was adapted from the DeepfakeBench training code,
# originally authored by Zhiyuan Yan (zhiyuanyan@link.cuhk.edu.cn)

# Original: https://github.com/SCLBD/DeepfakeBench/blob/main/training/train.py

import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        device,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()

        # create directory path
        self.log_dir = self.config['log_dir']
        print("Making dir ", self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        phase = phase.split('/')[-1]
        dataset_key = dataset_key.split('/')[-1]
        metric_key = metric_key.split('/')[-1]
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.device = self.device
        if 'cuda' in str(self.device) and self.config['ddp'] == True and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            self.model = DDP(self.model, device_ids=[self.config['gpu_id']],
                             find_unused_parameters=True, output_device=self.config['gpu_id'])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()

            return losses,predictions

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        validation_data_loaders=None
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        #times_per_epoch=4
        validation_step = len(train_data_loader) // times_per_epoch    # validate 10 times per epoch
        step_cnt = epoch * len(train_data_loader)
        
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            # if using GPU, move data to GPU
            if 'cuda' in str(self.device):
                for key in data_dict.keys():
                    if data_dict[key] is not None and key!='name':
                        data_dict[key] = data_dict[key].cuda()

            losses, predictions=self.train_step(data_dict)
            # update learning rate

            if self.config.get('SWA', False) and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['gpu_id']==0:
                if self.config.get('SWA', False) and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    processed_train_dataset = [dataset.split('/')[-1] for dataset in self.config['train_dataset']]
                    processed_train_dataset = ','.join(processed_train_dataset)
                    writer = self.get_writer('train', processed_train_dataset, k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    processed_train_dataset = [dataset.split('/')[-1] for dataset in self.config['train_dataset']]
                    processed_train_dataset = ','.join(processed_train_dataset)
                    writer = self.get_writer('train', processed_train_dataset, k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run validation
            if (step_cnt+1) % validation_step == 0:
                if validation_data_loaders is not None and ((not self.config['ddp']) or (self.config['ddp'] and dist.get_rank() == 0)):
                    self.logger.info("===> Validation start!")
                    validation_best_metric = self.eval(
                        eval_data_loaders=validation_data_loaders,
                        eval_stage="validation",
                        step=step_cnt,
                        epoch=epoch,
                        iteration=iteration
                    )
                else:
                    validation_best_metric = None

            step_cnt += 1

            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cpu()
        return validation_best_metric
    
    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def eval_one_dataset(self, data_loader):
        # define eval recorder
        eval_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # if using GPU, move data to GPU
            if 'cuda' in str(self.device):
                for key in data_dict.keys():
                    if data_dict[key] is not None:
                        data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict) #dict with keys cls, feat
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            # Get the predicted class for each sample in the batch
            _, predicted_classes = torch.max(predictions['cls'], dim=1)
            # Convert the predicted class indices to a list and add to prediction_lists
            prediction_lists += predicted_classes.cpu().detach().numpy().tolist()
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    eval_recorder_loss[name].update(value)
        return eval_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset,eval_stage):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if eval_stage=='validation' and self.config['save_ckpt']:
                self.save_ckpt(eval_stage, key, f"{epoch}+{iteration}")
            self.save_metrics(eval_stage, metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer(eval_stage, key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'{eval_stage}_losses/{k}', v_avg, global_step=step)
                loss_str += f"{eval_stage}-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"{eval_stage}-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer(eval_stage, key, k)
            writer.add_scalar(f'{eval_stage}_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'{eval_stage}-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'{eval_stage}_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'{eval_stage}_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def eval(self, eval_data_loaders, eval_stage, step=None, epoch=None, iteration=None):
        # set model to eval mode
        self.setEval()

        # define eval recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'dataset_dict':{}} #'video_auc': 0
        keys = eval_data_loaders.keys()
        for key in keys:
            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.eval_one_dataset(eval_data_loaders[key])
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps, logger=self.logger)
            
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"{eval_stage}-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset,eval_stage)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric, eval_stage)

        self.logger.info(f'===> {eval_stage} Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    
    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions