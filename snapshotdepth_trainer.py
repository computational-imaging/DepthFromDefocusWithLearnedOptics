import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
from snapshotdepth import SnapshotDepth
from util.log_manager import LogManager

seed_everything(123)


def prepare_data(hparams):
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    randcrop = hparams.randcrop

    padding = 0
    val_idx = 3994
    sf_train_dataset = SceneFlow('train',
                                 (image_sz + 4 * crop_width,
                                  image_sz + 4 * crop_width),
                                 is_training=True,
                                 randcrop=randcrop, augment=augment, padding=padding,
                                 singleplane=False)
    sf_train_dataset = torch.utils.data.Subset(sf_train_dataset,
                                               range(val_idx, len(sf_train_dataset)))

    sf_val_dataset = SceneFlow('train',
                               (image_sz + 4 * crop_width,
                                image_sz + 4 * crop_width),
                               is_training=False,
                               randcrop=randcrop, augment=augment, padding=padding,
                               singleplane=False)
    sf_val_dataset = torch.utils.data.Subset(sf_val_dataset, range(val_idx))

    if hparams.mix_dualpixel_dataset:
        dp_train_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)
        dp_val_dataset = DualPixel('val',
                                   (image_sz + 4 * crop_width,
                                    image_sz + 4 * crop_width),
                                   is_training=False,
                                   randcrop=randcrop, augment=augment, padding=padding)

        train_dataset = ConcatDataset([dp_train_dataset, sf_train_dataset])
        val_dataset = ConcatDataset([dp_val_dataset, sf_val_dataset])

        n_sf = len(sf_train_dataset)
        n_dp = len(dp_train_dataset)
        sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
                                    1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz, sampler=sampler,
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    else:
        train_dataset = sf_train_dataset
        val_dataset = sf_val_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def main(args):
    logger = TensorBoardLogger(args.default_root_dir,
                               name=args.experiment_name)

    logmanager_callback = LogManager()

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        filepath=os.path.join(logger.log_dir, 'checkpoints', '{epoch}-{val_loss:.4f}'),
        save_top_k=1,
        period=1,
        mode='min',
    )

    model = SnapshotDepth(hparams=args, log_dir=logger.log_dir)
    train_dataloader, val_dataloader = prepare_data(hparams=args)

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[logmanager_callback],
        checkpoint_callback=checkpoint_callback,
        sync_batchnorm=True,
        benchmark=True,
    )
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='LearnedDepth')
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dual_pixel_dataset', action='store_true')
    parser.set_defaults(mix_dualpixel_dataset=True)

    parser = Trainer.add_argparse_args(parser)
    parser = SnapshotDepth.add_model_specific_args(parser)

    parser.set_defaults(
        gpus=1,
        default_root_dir='data/logs',
        max_epochs=100,
    )

    args = parser.parse_args()

    main(args)
