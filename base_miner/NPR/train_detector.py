from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import random
import torch

from base_miner.NPR.validate import validate
from base_miner.NPR.networks.trainer import Trainer
from base_miner.config import IMAGE_DATASETS as DATASET_META
from base_miner.NPR.options import TrainOptions
from bitmind.utils.image_transforms import get_base_transforms, get_random_augmentations
from base_miner.datasets.util import load_and_split_datasets, create_real_fake_datasets


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main():
    opt = TrainOptions().parse()
    seed_torch(100)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    # RealFakeDataseta will limit the number of images sampled per dataset to the length of the smallest dataset
    base_transforms = get_base_transforms()
    random_augs = get_random_augmentations()
    split_transforms = {
        'train': random_augs,
        'validation': base_transforms,
        'test': base_transforms
    }
    real_datasets = load_and_split_datasets(
        DATASET_META['real'], modality='image', split_transforms=split_transforms)
    fake_datasets = load_and_split_datasets(
        DATASET_META['fake'], modality='image', split_transforms=split_transforms)
    train_dataset, val_dataset, test_dataset = create_real_fake_datasets(
        real_datasets, fake_datasets)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=lambda d: tuple(d))
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda d: tuple(d))
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda d: tuple(d))

    model = Trainer(opt)
    display_loss_steps = 10
    early_stopping_epochs = 10
    best_val_acc = 0
    n_epoch_since_improvement = 0
    model.train()

    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):

        for step, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            if step % display_loss_steps == 0:
                ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                print(f"{ts} | Step: {step} ({model.total_steps}) | Train loss: {model.loss} | lr {model.lr}")

            if model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            model.total_steps += 1

        if epoch % opt.delr_freq == 0 and epoch != 0:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            print(ts, 'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_loader)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)

        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        if acc > best_val_acc:
            model.save_networks('best')
            best_val_acc = acc
        else:
            n_epoch_since_improvement += 1
            if n_epoch_since_improvement >= early_stopping_epochs:
                break

        model.train()

    model.eval()
    acc, ap = validate(model.model, test_loader)[:2]
    print("(Test) acc: {}; ap: {}".format(acc, ap))
    model.save_networks('last')


if __name__ == '__main__':
    main()


