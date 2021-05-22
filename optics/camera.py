import abc
import math
from typing import List, Union

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import complex, cubicspline
from util.fft import fftshift
from util.helper import copy_quadruple, depthmap_to_layereddepth, heightmap_to_phase, ips_to_metric, over_op, \
    refractive_index


class BaseCamera(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, **kwargs):
        super().__init__()
        assert min_depth > 1e-6, f'Minimum depth is too small. min_depth: {min_depth}'
        scene_distances = ips_to_metric(torch.linspace(0, 1, steps=n_depths), min_depth, max_depth)

        self._register_wavlength(wavelengths)

        self.n_depths = len(scene_distances)
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.focal_depth = focal_depth
        self.mask_diameter = mask_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.focal_length = focal_length
        self.f_number = self.focal_length / self.mask_diameter
        self.image_size = self._normalize_image_size(image_size)
        self.mask_pitch = self.mask_diameter / mask_size
        self.mask_size = mask_size

        self.register_buffer('scene_distances', scene_distances)

        self.build_camera()

    def _register_wavlength(self, wavelengths):
        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths)  # in [meter]
        elif isinstance(wavelengths, float):
            wavelengths = torch.tensor([wavelengths])
        else:
            raise ValueError('wavelengths has to be a float or a list of floats.')

        if len(wavelengths) % 3 != 0:
            raise ValueError('the number of wavelengths has to be a multiple of 3.')

        self.n_wl = len(wavelengths)
        if not hasattr(self, 'wavelengths'):
            self.register_buffer('wavelengths', wavelengths)
        else:
            self.wavelengths = wavelengths.to(self.wavelengths.device)

    @abc.abstractmethod
    def build_camera(self):
        pass

    def sensor_distance(self):
        return 1. / (1. / self.focal_length - 1. / self.focal_depth)

    def normalize_psf(self, psfimg):
        # Scale the psf
        # As the incoming light doesn't change, we compute the PSF energy without the phase modulation
        # and use it to normalize PSF with phase modulation.
        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)

    def _capture_impl(self, volume, layered_depth, psf, occlusion, eps=1e-3):
        scale = volume.max()
        volume = volume / scale
        Fpsf = torch.rfft(psf, 2)

        if occlusion:
            Fvolume = torch.rfft(volume, 2)
            Flayered_depth = torch.rfft(layered_depth, 2)
            blurred_alpha_rgb = torch.irfft(
                complex.multiply(Flayered_depth, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_volume = torch.irfft(
                complex.multiply(Fvolume, Fpsf), 2, signal_sizes=volume.shape[-2:])

            # Normalize the blurred intensity
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
            Fcumsum_alpha = torch.rfft(cumsum_alpha, 2)
            blurred_cumsum_alpha = torch.irfft(
                complex.multiply(Fcumsum_alpha, Fpsf), 2, signal_sizes=volume.shape[-2:])
            blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
            blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)

            over_alpha = over_op(blurred_alpha_rgb)
            captimg = torch.sum(over_alpha * blurred_volume, dim=-3)
        else:
            Fvolume = torch.rfft(volume, 2)
            Fcaptimg = complex.multiply(Fvolume, Fpsf).sum(dim=2)
            captimg = torch.irfft(Fcaptimg, 2, signal_sizes=volume.shape[-2:])

        captimg = scale * captimg
        volume = scale * volume
        return captimg, volume

    def _capture_from_rgbd_with_psf_impl(self, img, depthmap, psf, occlusion):
        layered_depth = depthmap_to_layereddepth(depthmap, self.n_depths, binary=True)
        volume = layered_depth * img[:, :, None, ...]
        return self._capture_impl(volume, layered_depth, psf, occlusion)

    def capture_from_rgbd(self, img, depthmap, occlusion):
        psf = self.psf_at_camera(size=img.shape[-2:]).unsqueeze(0)  # add batch dimension
        psf = self.normalize_psf(psf)
        return self.capture_from_rgbd_with_psf(img, depthmap, psf, occlusion)

    def capture_from_rgbd_with_psf(self, img, depthmap, psf, occlusion):
        return self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)[0]

    @abc.abstractmethod
    def psf_at_camera(self, size=None, modulate_phase=True, is_training=torch.tensor(False)):
        pass

    @abc.abstractmethod
    def heightmap(self):
        pass

    def forward(self, img, depthmap, occlusion, is_training=torch.tensor(False)):
        """
        Args:
            img: B x C x H x W

        Returns:
            captured image: B x C x H x W
        """
        psf = self.psf_at_camera(size=img.shape[-2:], is_training=is_training).unsqueeze(0)  # add batch dimension
        psf = self.normalize_psf(psf)
        captimg, volume = self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    def _normalize_image_size(self, image_size):
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        elif isinstance(image_size, list):
            if image_size[0] % 2 == 1 or image_size[1] % 2 == 1:
                raise ValueError('Image size has to be even.')
        else:
            raise ValueError('image_size has to be int or list of int.')
        return image_size

    def set_image_size(self, image_size):
        image_size = self._normalize_image_size(image_size)
        self.image_size = image_size
        self.build_camera()

    def set_wavelengths(self, wavelengths):
        self._register_wavlength(wavelengths)
        self.build_camera()

    def set_n_depths(self, n_depths):
        self.n_depths = n_depths
        self.build_camera()

    def extra_repr(self):
        msg = f'Camera module...\n' \
              f'Refcative index for center wavelength: {refractive_index(self.wavelengths[self.n_wl // 2])} \n' \
              f'Mask pitch: {self.mask_pitch * 1e6}[um] \n' \
              f'f number: {self.f_number} \n' \
              f'Depths: {self.scene_distances} \n' \
              f'Input image size: {self.image_size} \n'
        return msg


class BaseRotationallySymmetricCamera(BaseCamera):

    def __init__(self, focal_depth: float, min_depth: float, max_depth: float, n_depths: int,
                 image_size: Union[int, List[int]], mask_size: int, focal_length: float, mask_diameter: float,
                 camera_pixel_pitch: float, wavelengths: List[float], full_size=1920):
        self.full_size = self._normalize_image_size(full_size)
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                         focal_length, mask_diameter, camera_pixel_pitch, wavelengths)

    def build_camera(self):
        prop_amplitude, prop_phase = self.pointsource_inputfield1d()

        H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        ind = self.find_index(rho_grid, rho_sampling)

        H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
        ind_full = self.find_index(rho_grid_full, rho_sampling_full)

        assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
            'Grid (max): {}, Sampling (max): {}'.format(
                rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
        assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
            'Grid (min): {}, Sampling (min): {}'.format(
                rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])

        self.register_buffer('prop_amplitude', prop_amplitude)
        self.register_buffer('prop_phase', prop_phase)

        self.register_buffer('H', H)
        self.register_buffer('rho_grid', rho_grid)
        self.register_buffer('rho_sampling', rho_sampling)
        self.register_buffer('ind', ind)

        self.register_buffer('H_full', H_full)
        self.register_buffer('rho_grid_full', rho_grid_full)
        # These two parameters are not used for training.
        self.rho_sampling_full = rho_sampling_full
        self.ind_full = ind_full

    @abc.abstractmethod
    def heightmap1d(self):
        pass

    def find_index(self, a, v):
        a = a.squeeze(1).cpu().numpy()
        v = v.cpu().numpy()
        index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
        return torch.from_numpy(index)

    def heightmap(self):
        heightmap1d = torch.cat([self.heightmap1d().cpu(), torch.zeros((self.mask_size // 2))], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.mask_size, dtype=torch.double)
        y_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord)
        heightmap11 = cubicspline.interp(r_grid, heightmap1d, r_coord, ind).float()
        heightmap = copy_quadruple(heightmap11).squeeze()
        return heightmap

    def pointsource_inputfield1d(self):
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        # compute pupil function
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        scene_distances = self.scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths

        radius = torch.sqrt(scene_distances ** 2 + r ** 2)  # 1 x D x n_r

        # ignore 1/j term (constant phase)
        amplitude = scene_distances / wavelengths / radius ** 2  # n_wl x D x n_r
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (radius - scene_distances)  # n_wl x D x n_r
        if not math.isinf(self.focal_depth):
            focal_depth = torch.tensor(self.focal_depth).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase

    def precompute_H(self, image_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.camera_pixel_pitch * torch.arange(1, image_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.camera_pixel_pitch * torch.arange(1, image_size[1] // 2 + 1).reshape(1, -1)
        coord_y = coord_y.double()
        coord_x = coord_x.double()
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                torch.arange(-1, max(image_size) // 2 + 1, dtype=torch.double) + 0.5)

        # n_wl x 1 x n_rho_grid
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())

        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2).double()
        r = r.reshape(1, -1, 1)
        J = torch.where(rho_grid == 0,
                        1 / 2 * r ** 2,
                        1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r))
        h = J[:, 1:, :] - J[:, :-1, :]
        h0 = J[:, 0:1, :]
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling

    def psf1d(self, H, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        H = H.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            phase_delays = heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                              refractive_index(wavelengths))

            phase = phase_delays + self.prop_phase  # n_wl X D x n_r
        else:
            phase = self.prop_phase

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = self.prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), H).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), H).squeeze(-2)

        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)  # n_wl X D X n_rho

    def psf_out_of_fov_energy(self, psf_size: int):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d_full()
        edge = psf_size / 2 * self.camera_pixel_pitch / (
                self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        psf1d_out_of_fov = psf1d * (self.rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()

    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, modulate_phase)
        psf_rd = F.relu(cubicspline.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return copy_quadruple(psf_rd)

    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), **kwargs):
        psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, modulate_phase)
        if size is not None:
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))


class RotationallySymmetricCamera(BaseRotationallySymmetricCamera):

    def __init__(self, focal_depth: float, min_depth: float, max_depth: float, n_depths: int,
                 image_size: Union[int, List[int]], mask_size: int, focal_length: float, mask_diameter: float,
                 camera_pixel_pitch: float, wavelengths=[632e-9, 550e-9, 450e-9], full_size=1920,
                 mask_upsample_factor=1, requires_grad: bool = False):
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                         focal_length, mask_diameter, camera_pixel_pitch, wavelengths, full_size)
        init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)  # 1D half size (radius)
        self.heightmap1d_ = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.mask_upsample_factor = mask_upsample_factor

    def heightmap1d(self):
        return F.interpolate(self.heightmap1d_.reshape(1, 1, -1),
                             scale_factor=self.mask_upsample_factor, mode='nearest').reshape(-1)

    def psf1d_full(self):
        return self.psf1d(self.H_full, modulate_phase=torch.tensor(True)) / self.normalization_scaler.squeeze(-1)

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion)


class MixedCamera(RotationallySymmetricCamera):

    def __init__(self, focal_depth: float, min_depth: float, max_depth: float, n_depths: int,
                 image_size: Union[int, List[int]], mask_size: int, focal_length: float, mask_diameter: float,
                 camera_pixel_pitch: float, wavelengths=[632e-9, 550e-9, 450e-9], mask_upsample_factor=1,
                 diffraction_efficiency=0.7, full_size=100, requires_grad: bool = False):
        self.diffraction_efficiency = diffraction_efficiency
        super().__init__(focal_depth, min_depth, max_depth, n_depths, image_size, mask_size, focal_length,
                         mask_diameter, camera_pixel_pitch, wavelengths, full_size, mask_upsample_factor,
                         requires_grad)

    def build_camera(self):
        H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        ind = self.find_index(rho_grid, rho_sampling)

        H_full, rho_grid_full, rho_sampling_full = self.precompute_H(self.full_size)
        ind_full = self.find_index(rho_grid_full, rho_sampling_full)

        assert (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0]).all(), \
            'Grid (max): {}, Sampling (max): {}'.format(
                rho_grid.max(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).max(dim=-1)[0])
        assert (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0]).all(), \
            'Grid (min): {}, Sampling (min): {}'.format(
                rho_grid.min(dim=-1)[0], rho_sampling.reshape(self.n_wl, -1).min(dim=-1)[0])
        self.register_buffer('H', H)
        self.register_buffer('rho_grid', rho_grid)
        self.register_buffer('rho_sampling', rho_sampling)
        self.register_buffer('ind', ind)
        self.register_buffer('H_full', H_full)
        self.register_buffer('rho_grid_full', rho_grid_full)
        # These two parameters are not used for training.
        self.rho_sampling_full = rho_sampling_full
        self.ind_full = ind_full

    def pointsource_inputfield1d(self, scene_distances):
        device = scene_distances.device
        r = self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2, device=device).double()
        # compute pupil function
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        scene_distances = scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths

        radius = torch.sqrt(scene_distances ** 2 + r ** 2)  # 1 x D x n_r

        # ignore 1/j term (constant phase)
        amplitude = scene_distances / wavelengths / radius ** 2  # n_wl x D x n_r
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (radius - scene_distances)  # n_wl x D x n_r
        if not math.isinf(self.focal_depth):
            focal_depth = torch.tensor(self.focal_depth, device=device).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase

    def psf1d(self, H, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(scene_distances)

        H = H.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        if modulate_phase:
            phase_delays = heightmap_to_phase(self.heightmap1d().reshape(1, -1),  # add wavelength dim
                                              wavelengths,
                                              refractive_index(wavelengths))
            phase = phase_delays + prop_phase  # n_wl X D x n_r
        else:
            phase = prop_phase

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), H).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), H).squeeze(-2)

        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)  # n_wl X D X n_rho

    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.psf1d(H, scene_distances, modulate_phase)
        psf_rd = F.relu(cubicspline.interp(rho_grid, psf1d, rho_sampling, ind).float())
        psf_rd = psf_rd.reshape(self.n_wl, self.n_depths, size[0] // 2, size[1] // 2)
        return copy_quadruple(psf_rd)

    def psf_at_camera(self, size=None, modulate_phase=torch.tensor(True), is_training=torch.tensor(False)):
        device = self.H.device
        if is_training:
            scene_distances = ips_to_metric(
                torch.linspace(0, 1, steps=self.n_depths, device=device) +
                1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                self.min_depth, self.max_depth)
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
        else:
            scene_distances = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                            self.min_depth, self.max_depth)

        diffracted_psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, modulate_phase)
        undiffracted_psf = self._psf_at_camera_impl(
            self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances, torch.tensor(False))

        # Keep the normalization factor for penalty computation
        self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

        diffracted_psf = diffracted_psf / self.diff_normalization_scaler
        undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

        psf = self.diffraction_efficiency * diffracted_psf + (1 - self.diffraction_efficiency) * undiffracted_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if torch.tensor(size is not None):
            pad_h = (size[0] - self.image_size[0]) // 2
            pad_w = (size[1] - self.image_size[1]) // 2
            psf = F.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fftshift(psf, dims=(-1, -2))

    def psf_out_of_fov_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        device = self.H.device
        scene_distances = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                        self.min_depth, self.max_depth)
        psf1d_diffracted = self.psf1d_full(scene_distances, torch.tensor(True))
        # Normalize PSF based on the cropped PSF
        psf1d_diffracted = psf1d_diffracted / self.diff_normalization_scaler.squeeze(-1)
        edge = psf_size / 2 * self.camera_pixel_pitch / (
                self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance())
        psf1d_out_of_fov = psf1d_diffracted * (self.rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()

    def psf1d_full(self, scene_distances, modulate_phase=torch.tensor(True)):
        return self.psf1d(self.H_full, scene_distances, modulate_phase=modulate_phase)

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion, is_training=torch.tensor(True))

    def set_diffraction_efficiency(self, de: float):
        self.diffraction_efficiency = de
        self.build_camera()
