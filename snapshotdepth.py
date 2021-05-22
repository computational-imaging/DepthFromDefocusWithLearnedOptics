"""
python snapshotdepth_trainer.py --gpus 4 --experiment_name 'fabrication_mixed_camera' --occlusion --randcrop --augment --batch_sz 3 --preinverse --camera_type mixed --optimize_optics --bayer --focal_depth 1.7 --distributed_backend ddp  --max_epochs 1000 --psf_loss_weight 1.00

"""
import copy
from argparse import ArgumentParser
from collections import namedtuple

import pytorch_lightning as pl
import torch
import torch.optim
import torchvision.transforms
import torchvision.utils
from debayer import Debayer3x3
from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError

from models.simple_model import SimpleModel
from optics import camera
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from util.loss import Vgg16PerceptualLoss

SnapshotOutputs = namedtuple('SnapshotOutputs',
                             field_names=['captimgs', 'captimgs_linear',
                                          'est_images', 'est_depthmaps',
                                          'target_images', 'target_depthmaps',
                                          'psf'])


class SnapshotDepth(pl.LightningModule):

    def __init__(self, hparams, log_dir=None):
        super().__init__()

        self.hparams = copy.deepcopy(hparams)
        self.save_hyperparameters(self.hparams)

        self.__build_model()

        self.metrics = {
            'depth_loss': MeanAbsoluteError(),
            'image_loss': MeanAbsoluteError(),
            'mae_depthmap': MeanAbsoluteError(),
            'mse_depthmap': MeanSquaredError(),
            'mae_image': MeanAbsoluteError(),
            'mse_image': MeanSquaredError(),
            'vgg_image': MeanSquaredError(),
        }

        self.log_dir = log_dir

    def set_image_size(self, image_sz):
        self.hparams.image_sz = image_sz
        if type(image_sz) == int:
            image_sz += 4 * self.crop_width
        else:
            image_sz[0] += 4 * self.crop_width
            image_sz[1] += 4 * self.crop_width

        self.camera.set_image_size(image_sz)

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 4000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 4000.)
            optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.optics_lr
            optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.cnn_lr
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        params = [
            {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer

    def training_step(self, samples, batch_idx):
        target_images = samples['image']
        target_depthmaps = samples['depthmap']
        depth_conf = samples['depth_conf']

        if depth_conf.ndim == 4:
            depth_conf = crop_boundary(depth_conf, self.crop_width * 2)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

        # Unpack outputs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps
        target_images = outputs.target_images
        target_depthmaps = outputs.target_depthmaps
        captimgs_linear = outputs.captimgs_linear

        data_loss, loss_logs = self.__compute_loss(outputs, target_depthmaps, target_images, depth_conf)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        misc_logs = {
            'train_misc/target_depth_max': target_depthmaps.max(),
            'train_misc/target_depth_min': target_depthmaps.min(),
            'train_misc/est_depth_max': est_depthmaps.max(),
            'train_misc/est_depth_min': est_depthmaps.min(),
            'train_misc/target_image_max': target_images.max(),
            'train_misc/target_image_min': target_images.min(),
            'train_misc/est_image_max': est_images.max(),
            'train_misc/est_image_min': est_images.min(),
            'train_misc/captimg_max': captimgs_linear.max(),
            'train_misc/captimg_min': captimgs_linear.min(),
        }
        if self.hparams.optimize_optics:
            misc_logs.update({
                'optics/heightmap_max': self.camera.heightmap1d().max(),
                'optics/heightmap_min': self.camera.heightmap1d().min(),
                'optics/psf_out_of_fov_energy': loss_logs['train_loss/psf_loss'],
                'optics/psf_out_of_fov_max': loss_logs['train_loss/psf_out_of_fov_max'],
            })

        logs = {}
        logs.update(loss_logs)
        logs.update(misc_logs)

        if not self.global_step % self.hparams.summary_track_train_every:
            self.__log_images(outputs, target_images, target_depthmaps, 'train')

        self.log_dict(logs)

        return data_loss

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_step(self, samples, batch_idx):
        target_images = samples['image']
        target_depthmaps = samples['depthmap']
        depth_conf = samples['depth_conf']
        if depth_conf.ndim == 4:
            depth_conf = crop_boundary(depth_conf, 2 * self.crop_width)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

        # Unpack outputs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps
        target_images = outputs.target_images
        target_depthmaps = outputs.target_depthmaps

        est_depthmaps = est_depthmaps * depth_conf
        target_depthmaps = target_depthmaps * depth_conf
        self.metrics['mae_depthmap'](est_depthmaps, target_depthmaps)
        self.metrics['mse_depthmap'](est_depthmaps, target_depthmaps)
        self.metrics['mae_image'](est_images, target_images)
        self.metrics['mse_image'](est_images, target_images)
        self.metrics['vgg_image'](est_images, target_images)

        self.log('validation/mse_depthmap', self.metrics['mse_depthmap'], on_step=False, on_epoch=True)
        self.log('validation/mae_depthmap', self.metrics['mae_depthmap'], on_step=False, on_epoch=True)
        self.log('validation/mse_image', self.metrics['mse_image'], on_step=False, on_epoch=True)
        self.log('validation/mae_image', self.metrics['mae_image'], on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.__log_images(outputs, target_images, target_depthmaps, 'validation')


    def validation_epoch_end(self, outputs):
        val_loss = self.__combine_loss(self.metrics['mae_depthmap'].compute(),
                                       self.metrics['vgg_image'].compute(),
                                       0.)
        self.log('val_loss', val_loss)

    def forward(self, images, depthmaps, is_testing):
        # invert the gamma correction for sRGB image
        images_linear = srgb_to_linear(images)

        # Currently PSF jittering is supported only for MixedCamera.
        if torch.tensor(self.hparams.psf_jitter):
            # Jitter the PSF on the evaluation as well.
            captimgs, target_volumes, _ = self.camera.forward_train(images_linear, depthmaps,
                                                                    occlusion=self.hparams.occlusion)
            # We don't want to use the jittered PSF for the pseudo inverse.
            psf = self.camera.psf_at_camera(is_training=torch.tensor(False)).unsqueeze(0)
        else:
            captimgs, target_volumes, psf = self.camera.forward(images_linear, depthmaps,
                                                                occlusion=self.hparams.occlusion)

        # add some Gaussian noise
        dtype = images.dtype
        device = images.device
        noise_sigma_min = self.hparams.noise_sigma_min
        noise_sigma_max = self.hparams.noise_sigma_max
        noise_sigma = (noise_sigma_max - noise_sigma_min) * torch.rand((captimgs.shape[0], 1, 1, 1), device=device,
                                                                       dtype=dtype) + noise_sigma_min

        # without Bayer
        if not torch.tensor(self.hparams.bayer):
            captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=device, dtype=dtype)
        else:
            captimgs_bayer = to_bayer(captimgs)
            captimgs_bayer = captimgs_bayer + noise_sigma * torch.randn(captimgs_bayer.shape, device=device,
                                                                        dtype=dtype)
            captimgs = self.debayer(captimgs_bayer)

        # Crop the boundary artifact of DFT-based convolution
        captimgs = crop_boundary(captimgs, self.crop_width)
        target_volumes = crop_boundary(target_volumes, self.crop_width)

        if self.hparams.preinverse:
            # Apply the Tikhonov-regularized inverse
            psf_cropped = crop_psf(psf, captimgs.shape[-2:])
            pinv_volumes = apply_tikhonov_inverse(captimgs, psf_cropped, self.hparams.reg_tikhonov,
                                                  apply_edgetaper=True)
        else:
            pinv_volumes = torch.zeros_like(target_volumes)

        # Feed the cropped images to CNN
        model_outputs = self.decoder(captimgs=captimgs,
                                     pinv_volumes=pinv_volumes,
                                     images=images_linear,
                                     depthmaps=depthmaps)

        # Require twice cropping because the image formation also crops the boundary.
        target_images = crop_boundary(images, 2 * self.crop_width)
        target_depthmaps = crop_boundary(depthmaps, 2 * self.crop_width)

        captimgs = crop_boundary(captimgs, self.crop_width)
        est_images = crop_boundary(model_outputs.est_images, self.crop_width)
        est_depthmaps = crop_boundary(model_outputs.est_depthmaps, self.crop_width)

        outputs = SnapshotOutputs(
            target_images=target_images,
            target_depthmaps=target_depthmaps,
            captimgs=linear_to_srgb(captimgs),
            captimgs_linear=captimgs,
            est_images=est_images,
            est_depthmaps=est_depthmaps,
            psf=psf,
        )

        return outputs

    def __build_model(self):
        hparams = self.hparams
        self.crop_width = hparams.crop_width
        mask_diameter = hparams.focal_length / hparams.f_number

        wavelengths = [632e-9, 550e-9, 450e-9]
        camera_recipe = {
            'wavelengths': wavelengths,
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz + 4 * self.crop_width,
            'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'mask_diameter': mask_diameter,
            'mask_size': hparams.mask_sz,
        }
        optimize_optics = hparams.optimize_optics

        camera_recipe['mask_upsample_factor'] = hparams.mask_upsample_factor
        camera_recipe['diffraction_efficiency'] = hparams.diffraction_efficiency
        camera_recipe['full_size'] = hparams.full_size
        self.camera = camera.MixedCamera(**camera_recipe, requires_grad=optimize_optics)

        self.decoder = SimpleModel(hparams)
        self.debayer = Debayer3x3()

        self.image_lossfn = Vgg16PerceptualLoss()
        self.depth_lossfn = torch.nn.L1Loss()

        print(self.camera)

    def __combine_loss(self, depth_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
               self.hparams.image_loss_weight * image_loss + \
               self.hparams.psf_loss_weight * psf_loss

    def __compute_loss(self, outputs, target_depthmaps, target_images, depth_conf):
        hparams = self.hparams
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps

        depth_loss = self.depth_lossfn(est_depthmaps * depth_conf, target_depthmaps * depth_conf)
        image_loss = self.image_lossfn.train_loss(est_images, target_images)

        psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(hparams.psf_size)
        psf_loss = psf_out_of_fov_sum

        total_loss = self.__combine_loss(depth_loss, image_loss, psf_loss)
        logs = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'image_loss': image_loss,
            'psf_loss': psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max,
        }
        return total_loss, logs

    @torch.no_grad()
    def __log_images(self, outputs, target_images, target_depthmaps, tag: str):
        # Unpack outputs
        captimgs = outputs.captimgs
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps

        summary_image_sz = self.hparams.summary_image_sz
        # CAUTION! Summary image is clamped, and visualized in sRGB.
        summary_max_images = min(self.hparams.summary_max_images, captimgs.shape[0])
        captimgs, target_images, target_depthmaps, est_images, est_depthmaps = [
            imresize(x, summary_image_sz)
            for x in [captimgs, target_images, target_depthmaps, est_images, est_depthmaps]
        ]
        target_depthmaps = gray_to_rgb(1.0 - target_depthmaps)
        est_depthmaps = gray_to_rgb(1.0 - est_depthmaps)  # Flip [0, 1] for visualization purpose
        summary = torch.cat([captimgs, target_images, est_images, target_depthmaps, est_depthmaps], dim=-2)
        summary = summary[:summary_max_images]
        grid_summary = torchvision.utils.make_grid(summary, nrow=summary_max_images)
        self.logger.experiment.add_image(f'{tag}/summary', grid_summary, self.global_step)

        if self.hparams.optimize_optics or self.global_step == 0:
            # PSF and heightmap is not visualized at computed size.
            psf = self.camera.psf_at_camera(size=(128, 128), is_training=torch.tensor(False))
            psf = self.camera.normalize_psf(psf)
            psf = fftshift(crop_psf(psf, 64), dims=(-1, -2))
            psf /= psf.max()
            heightmap = imresize(self.camera.heightmap()[None, None, ...],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            heightmap -= heightmap.min()
            heightmap /= heightmap.max()
            grid_psf = torchvision.utils.make_grid(psf[:, ::self.hparams.summary_depth_every].transpose(0, 1),
                                                   nrow=3, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf', grid_psf, self.global_step)
            self.logger.experiment.add_image('optics/heightmap', heightmap, self.global_step)

            psf /= psf.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
            grid_psf = torchvision.utils.make_grid(psf.transpose(0, 1),
                                                   nrow=3, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_stretched', grid_psf, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logger parameters
        parser.add_argument('--summary_max_images', type=int, default=4)
        parser.add_argument('--summary_image_sz', type=int, default=256)
        parser.add_argument('--summary_mask_sz', type=int, default=256)
        parser.add_argument('--summary_depth_every', type=int, default=1)
        parser.add_argument('--summary_track_train_every', type=int, default=4000)

        # training parameters
        parser.add_argument('--cnn_lr', type=float, default=1e-3)
        parser.add_argument('--optics_lr', type=float, default=1e-9)
        parser.add_argument('--batch_sz', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--randcrop', default=False, action='store_true')
        parser.add_argument('--augment', default=False, action='store_true')

        # loss parameters
        parser.add_argument('--depth_loss_weight', type=float, default=1.0)
        parser.add_argument('--image_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_size', type=int, default=64)

        # dataset parameters
        parser.add_argument('--image_sz', type=int, default=256)
        parser.add_argument('--n_depths', type=int, default=16)
        parser.add_argument('--min_depth', type=float, default=1.0)
        parser.add_argument('--max_depth', type=float, default=5.0)
        parser.add_argument('--crop_width', type=int, default=32)

        # solver parameters
        parser.add_argument('--reg_tikhonov', type=float, default=1.0)
        parser.add_argument('--model_base_ch', type=int, default=32)

        parser.add_argument('--preinverse', dest='preinverse', action='store_true')
        parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
        parser.set_defaults(preinverse=True)

        # optics parameters
        parser.add_argument('--camera_type', type=str, default='mixed')
        parser.add_argument('--mask_sz', type=int, default=8000)
        parser.add_argument('--focal_length', type=float, default=50e-3)
        parser.add_argument('--focal_depth', type=float, default=1.7)
        parser.add_argument('--f_number', type=float, default=6.3)
        parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--full_size', type=int, default=1920)
        parser.add_argument('--mask_upsample_factor', type=int, default=10)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)

        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=True)
        parser.add_argument('--occlusion', dest='occlusion', action='store_true')
        parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
        parser.set_defaults(occlusion=True)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=False)

        # model parameters
        parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
        parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
        parser.set_defaults(psf_jitter=True)

        return parser
