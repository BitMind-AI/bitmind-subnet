from tensorboardX import SummaryWriter
from validate import validate
from networks.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import random
import torch

from bitmind.image_transforms import base_transforms, random_aug_transforms
from util.data import load_datasets, create_real_fake_datasets
from options import TrainOptions


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
    real_datasets, fake_datasets = load_datasets()
    train_dataset, val_dataset, test_dataset = create_real_fake_datasets(
        real_datasets,
        fake_datasets,
        base_transforms,
        train_transforms=random_aug_transforms,
        val_transforms=base_transforms,
        test_transforms=base_transforms)

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


