from pathlib import Path

import numpy as np
import pytest
import torch
from colorama import Fore, Style
from hydra import compose, initialize
from torch.utils.data import random_split
from tqdm import tqdm

from snraware.projects.loss.imaging_loss import Combined_Loss
from snraware.projects.mri.denoising.data import MRIDenoisingDataset, MRIDenoisingDatasetTest
from snraware.projects.mri.denoising.model import DenoisingModel
from snraware.components.setup import end_timer, set_seed, start_timer


# -----------------------------------------------------------------
class TestSNRNoise:
    data_root = None
    test_root = None
    output_dir = None

    def setup_class(self):
        test_path = Path(__file__).parents[0].resolve()
        self.data_root = str(test_path / "data/mri/denoising/tra")
        self.test_root = str(test_path / "data/mri/denoising/test")

        self.output_dir = str(test_path / "../.run/output")

        set_seed(23234325)

    def teardown_class(self):
        pass

    # ---------------------------------------------------------------

    def test_train_data(self):
        """Test the training dataset class."""
        rng = np.random.default_rng(23234325)

        H, W = 64, 64
        T = 256
        rep = 8

        tra_data = MRIDenoisingDataset(
            data_dir=self.data_root,
            repetition=rep,
            cutout_shape=[H, W, T],
            min_noise_level=1.0,
            max_noise_level=32.0,
            only_white_noise=False,
            ignore_gmap=False,
            add_noise=True,
            add_salt_pepper=False,
            salt_pepper_amount=0.4,
            salt_pepper_prob=0.4,
            add_poisson=False,
            poisson_prob=0.4,
            shuffle_along_3rd_dim=False,
            shuffle_along_3rd_dim_prob=0.5,
            matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5],
            matrix_size_adjust_prob=0.0,
            resolution_reduction_prob=0.0,
            partial_fourier_prob=0.0,
            single_frame_mode=False,
            single_frame_mode_prob=1.0,
            dicom_mode=False,
            rng=rng,
        )

        GT = [18, 534]

        for k in GT:
            noisy_im, clean_im, noise_sigmas, noise_sigmas_generated = tra_data[k]

            assert noisy_im.shape == (rep, 3, T, H, W)
            assert clean_im.shape == (rep, 2, T, H, W)
            assert noise_sigmas.shape == (rep,)

            noisy_im = np.transpose(noisy_im.numpy(), (3, 4, 1, 2, 0))
            clean_im = np.transpose(clean_im.numpy(), (3, 4, 1, 2, 0))

            gmap = noisy_im[:, :, 2, :, :]
            noisy_im = noisy_im[:, :, 0, :, :] + 1j * noisy_im[:, :, 1, :, :]
            clean_im = clean_im[:, :, 0, :, :] + 1j * clean_im[:, :, 1, :, :]
            noise_sigmas = noise_sigmas.numpy()
            noise_sigmas_generated = noise_sigmas_generated.numpy()

            noise = (noisy_im - clean_im / np.sqrt(noise_sigmas_generated * noise_sigmas_generated + 1)) / gmap
            std_r = np.mean(np.std(np.real(noise), axis=2))
            std_i = np.mean(np.std(np.imag(noise), axis=2))

            print(
                f"Testing key {k}, noisy_im {np.linalg.norm(noisy_im)}, clean_im {np.linalg.norm(clean_im)}, noise_sigmas {np.linalg.norm(noise_sigmas)}, gmap {np.mean(gmap)}, noise_std {std_r} + {std_i}j"
            )

            assert abs(std_r - 1) < 0.3, f"noise std real for key {k}"
            assert abs(std_i - 1) < 0.3, f"noise std imag for key {k}"

    # ---------------------------------------------------------------

    @pytest.mark.parametrize(
        "cutout_shape",
        [[64, 64, 16], [32, 64, 12], [16, 32, 8]],
    )
    @pytest.mark.parametrize(
        "repetition",
        [1, 3],
    )
    @pytest.mark.parametrize(
        "dicom_mode",
        [True, False],
    )
    @pytest.mark.parametrize(
        "only_white_noise",
        [True, False],
    )
    @pytest.mark.parametrize(
        "ignore_gmap",
        [True, False],
    )
    @pytest.mark.parametrize(
        "add_noise",
        [True, False],
    )
    def test_train_data_augmentations(
        self, cutout_shape, repetition, dicom_mode, only_white_noise, ignore_gmap, add_noise
    ):
        """Test the training dataset class with different augmentations."""
        rng = np.random.default_rng(4569867)

        tra_data = MRIDenoisingDataset(
            data_dir=self.data_root,
            repetition=repetition,
            cutout_shape=cutout_shape,
            min_noise_level=1.0,
            max_noise_level=32.0,
            only_white_noise=only_white_noise,
            ignore_gmap=ignore_gmap,
            add_noise=add_noise,
            add_salt_pepper=True,
            salt_pepper_amount=0.4,
            salt_pepper_prob=0.4,
            add_poisson=True,
            poisson_prob=0.4,
            shuffle_along_3rd_dim=True,
            shuffle_along_3rd_dim_prob=0.05,
            matrix_size_adjust_ratio=[0.5, 0.75, 1.0, 1.25, 1.5],
            matrix_size_adjust_prob=0.2,
            resolution_reduction_prob=0.2,
            partial_fourier_prob=0.2,
            single_frame_mode=True,
            single_frame_mode_prob=0.05,
            dicom_mode=dicom_mode,
            rng=rng,
        )

        for _ in tqdm(
            range(3),
            desc=f"cutout_shape {cutout_shape}, repetition {repetition}, dicom_mode {dicom_mode}, only_white_noise {only_white_noise}, ignore_gmap {ignore_gmap}, add_noise {add_noise}",
        ):
            noisy_im, clean_im, noise_sigmas, noise_sigmas_generated = tra_data[rng.integers(0, len(tra_data))]

            assert noisy_im.shape == (repetition, 3, cutout_shape[2], cutout_shape[0], cutout_shape[1])
            assert clean_im.shape == (repetition, 2, cutout_shape[2], cutout_shape[0], cutout_shape[1])
            assert noise_sigmas.shape[0] == repetition
            assert noise_sigmas_generated.shape[0] == repetition

    # ---------------------------------------------------------------

    @pytest.mark.parametrize(
        "batch_size",
        [1, 4, 8],
    )
    @pytest.mark.parametrize(
        "repetition",
        [1, 3],
    )
    def test_train_dataloader(self, batch_size, repetition):
        rng = np.random.default_rng(4569867)

        tra_data = MRIDenoisingDataset(
            data_dir=self.data_root, repetition=repetition, cutout_shape=[64, 64, 16], rng=rng
        )

        val_size = int(len(tra_data) * 0.1)
        train_size = len(tra_data) - val_size
        train_set, val_set = random_split(tra_data, [train_size, val_size], generator=torch.Generator())

        tra_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, prefetch_factor=2
        )

        tra_iter = iter(tra_loader)

        for _ in tqdm(range(10), desc=f"batch_size {batch_size}, repetition {repetition}"):
            noisy_im, clean_im, noise_sigmas, noise_sigmas_generated = next(tra_iter, (None, None, None, None))

            if noisy_im is not None:
                assert noisy_im.shape == (batch_size, repetition, 3, 16, 64, 64)
            if clean_im is not None:
                assert clean_im.shape == (batch_size, repetition, 2, 16, 64, 64)
            if noise_sigmas is not None:
                assert noise_sigmas.shape == (batch_size, repetition)
            if noise_sigmas_generated is not None:
                assert noise_sigmas_generated.shape == (batch_size, repetition)

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, prefetch_factor=2
        )

        val_iter = iter(val_loader)

        for _ in tqdm(range(5), desc=f"batch_size {batch_size}, repetition {repetition}"):
            noisy_im, clean_im, noise_sigmas, noise_sigmas_generated = next(val_iter, (None, None, None, None))

            if noisy_im is not None:
                assert noisy_im.shape == (batch_size, repetition, 3, 16, 64, 64)
            if clean_im is not None:
                assert clean_im.shape == (batch_size, repetition, 2, 16, 64, 64)
            if noise_sigmas is not None:
                assert noise_sigmas.shape == (batch_size, repetition)
            if noise_sigmas_generated is not None:
                assert noise_sigmas_generated.shape == (batch_size, repetition)

    # ---------------------------------------------------------------

    def test_test_dataloader(self):
        # test_data = MRIDenoisingDatasetTest(data_dir=self.test_root)
        test_data = MRIDenoisingDatasetTest(data_dir="/fastdata/denoising/test")

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True, prefetch_factor=2
        )

        test_iter = iter(test_loader)

        for k in range(5):
            noisy_im, clean_im, noise_sigmas, noise_sigmas_generated = next(test_iter, (None, None, None, None))

            print(f"{noisy_im.shape}, {clean_im.shape}, {noise_sigmas.shape}, {noise_sigmas_generated.shape}")

            assert noisy_im.shape[-2] >= 128 and noisy_im.shape[-1] >= 128 and noisy_im.shape[1] == 3
            assert clean_im.shape[-2] >= 128 and clean_im.shape[-1] >= 128 and clean_im.shape[1] == 2
            assert noise_sigmas.shape == (1,)
            assert noise_sigmas_generated.shape == (1,)

            noisy_im = np.transpose(np.squeeze(noisy_im.numpy()), (2, 3, 1, 0))
            clean_im = np.transpose(np.squeeze(clean_im.numpy()), (2, 3, 1, 0))

            gmap = noisy_im[:, :, :, 2]
            noisy_im = noisy_im[:, :, :, 0] + 1j * noisy_im[:, :, :, 1]
            clean_im = clean_im[:, :, :, 0] + 1j * clean_im[:, :, :, 1]
            noise_sigmas = noise_sigmas.numpy()

            print(
                f"{np.linalg.norm(noisy_im)}, {np.linalg.norm(clean_im)}, {np.linalg.norm(noise_sigmas)}, {np.median(gmap)}"
            )

# ---------------------------------------------------------------

if __name__ == "__main__":
    t = TestSNRNoise()
    t.setup_class()
    t.test_test_dataloader()
    t.test_dataloader_speed(
        batch_size=4,
        repetition=1,
        num_workers=64,
        num_reads=16,
        add_noise=True,
        matrix_size_adjust_prob=0,
        only_white_noise=True,
        add_salt_pepper=False,
        add_poisson=False,
        shuffle_along_3rd_dim=False,
        single_frame_mode=False,
    )
    t.teardown_class()
