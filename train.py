import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import MODELS


import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = ArgumentParser(description='Training with DDP.')
    parser.add_argument('config',
                        type=str)
    parser.add_argument('gpus',
                        type=int)
    parser.add_argument('--work_dir',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--seed',
                        type=int,
                        default=1024)


    args = parser.parse_args()
    return args


def main():

    #global args
    # parse args
    args = parse_args()

    # parse cfg
    cfg = Config.fromfile(osp.join(f'configs/{args.config}.yaml'))


    # show information
    print(f'Now training with {args.config}...')

    # configure seed
    seed_everything(args.seed)

    # prepare data loader 在build_datasize里resize成（576，320）
    dataset = build_dataset(cfg.dataset)
    loader = DataLoader(dataset, cfg.imgs_per_gpu, shuffle=True, num_workers=cfg.workers_per_gpu, drop_last=True)
    '''
    subset_size = 50  # 每个epoch加载的样本数
    sampler = SubsetRandomSampler(range(len(dataset))[:subset_size])
    loader = DataLoader(dataset, batch_size=cfg.imgs_per_gpu, sampler=sampler, num_workers=cfg.workers_per_gpu, drop_last=True)
    '''
    
    #import ipdb
    #ipdb.set_trace()

    if cfg.model.name == 'depsam':
        cfg.data_link = dataset

    # define model
    model = MODELS.build(name=cfg.model.name, option=cfg)


    # define trainer
    work_dir = osp.join(args.work_dir, args.config)
    # save checkpoint every 'cfg.checkpoint_epoch_interval' epochs
    #import ipdb
    #ipdb.set_trace()
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          save_weights_only=True,
                                          save_top_k=-1,
                                          filename='checkpoint_{epoch}',
                                          every_n_epochs=cfg.checkpoint_epoch_interval)
    trainer = Trainer(accelerator='cuda',
                      default_root_dir=work_dir,
                      gpus=args.gpus,
                      num_nodes=1,
                      max_epochs=cfg.total_epochs,
                      callbacks=[checkpoint_callback])
    '''trainer = Trainer(accelerator='ddp',
                      default_root_dir=work_dir,
                      gpus=args.gpus,
                      num_nodes=1,
                      max_epochs=cfg.total_epochs,
                      callbacks=[checkpoint_callback])'''

    # training
    trainer.fit(model,loader)


if __name__ == '__main__':
    main()
