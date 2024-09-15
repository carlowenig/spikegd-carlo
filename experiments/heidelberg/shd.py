from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import h5py
import numpy as np
import torch

from spikegd.utils.online_dataset import OnlineDataset


@dataclass
class Speaker:
    index: int
    age: int
    gender: str
    body_height: float


@dataclass(frozen=True)
class SHDSample:
    times: torch.Tensor
    units: torch.Tensor
    label_index: int
    speaker_index: int
    audio_filename: str | None = None

    def __post_init__(self):
        assert self.times.shape == self.units.shape


def _get_dataset_arr(file: h5py.File, path: str, dtype=None):
    ds = file[path]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"Expected dataset at {path}, got {type(ds)}")
    arr = ds[:]

    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Expected dataaset slice to be a numpy array, got {type(arr)}"
        )

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


# Inspired by torch.datasets.MNIST
class SHD(OnlineDataset[SHDSample]):
    """Spiking Heidelberg digits (SHD) dataset."""

    mirrors = ["https://zenkelab.org/datasets/"]

    def __init__(
        self,
        data_root: str | Path,
        mode: Literal["train", "test"] = "train",
        download=True,
        verbose=False,
    ):
        self.mode = mode
        super().__init__(data_root=data_root, download=download, verbose=verbose)

        self._verbose_print("Loading data from h5 file")

        ### Data
        with h5py.File(self.path / f"shd_{mode}.h5", "r") as file:
            # times_arr and units_arr are arrays of numpy arrays
            self.times_arr = _get_dataset_arr(file, "spikes/times")
            self.units_arr = _get_dataset_arr(file, "spikes/units")
            self.label_index_arr = _get_dataset_arr(file, "labels", int)
            self.speaker_index_arr = _get_dataset_arr(file, "extra/speaker", int)

            # TODO: Rename to avoid confusion with config["Nsamples"] (ensemble size)
            self.Nsamples = len(self.times_arr)
            self.N = max(units.astype(int).max() for units in self.units_arr) + 1
            self.t_max = max(times.astype(float)[-1] for times in self.times_arr)

            assert (
                self.units_arr.shape
                == self.times_arr.shape
                == self.label_index_arr.shape
                == self.speaker_index_arr.shape
                == (self.Nsamples,)
            )

            # Store label names as array for easy access
            self.label_names = _get_dataset_arr(file, "extra/keys", str)
            self.Nlabel = len(self.label_names)

            # Store speaker info as arrays for easy access
            self.ages_by_speaker = _get_dataset_arr(file, "extra/meta_info/age", int)
            self.genders_by_speaker = _get_dataset_arr(
                file, "extra/meta_info/gender", str
            )
            self.body_heights_by_speaker = _get_dataset_arr(
                file, "extra/meta_info/body_height", float
            )

        self.age_arr = self.ages_by_speaker[self.speaker_index_arr]
        self.gender_arr = self.genders_by_speaker[self.speaker_index_arr]
        self.body_height_arr = self.body_heights_by_speaker[self.speaker_index_arr]

        # Derived quantities
        self.spike_count_arr = np.array([len(times) for times in self.times_arr])
        self.trial_length_arr = np.array(
            [times.astype(float)[-1] for times in self.times_arr]
        )
        self.spike_rate_arr = self.spike_count_arr / self.trial_length_arr

        self._verbose_print("Loading audio filenames")

        ### Audio
        self.audio_filenames = np.loadtxt(
            self.path / f"hd_audio/{mode}_filenames.txt",
            dtype=str,
            delimiter=None,
        )

        self._verbose_print("Finished loading SHD")

    @override
    def _download_from_current_mirror(self):
        # Readme
        self._download_file("README.md")

        # Data
        self._download_file("shd_train.h5.gz")
        self._extract_gzip("shd_train.h5.gz")

        self._download_file("shd_test.h5.gz")
        self._extract_gzip("shd_test.h5.gz")

        # Audio
        self._download_file("hd_audio.tar.gz")
        self._extract_gzip("hd_audio.tar.gz")
        self._extract_tar("hd_audio.tar", "hd_audio")

    @override
    def _expect_exists(self):
        self._expect_file("README.md")
        self._expect_file("shd_train.h5")
        self._expect_file("shd_test.h5")
        self._expect_dir("hd_audio/audio")
        self._expect_file("hd_audio/train_filenames.txt")
        self._expect_file("hd_audio/test_filenames.txt")

    @override
    def __getitem__(self, sample_index):
        return self.get_sample(sample_index)

    @override
    def __len__(self):
        return self.Nsamples

    def get_speaker(self, speaker_index: int):
        return Speaker(
            index=speaker_index,
            age=int(self.ages_by_speaker[speaker_index]),
            gender=str(self.genders_by_speaker[speaker_index]),
            body_height=float(self.body_heights_by_speaker[speaker_index]),
        )

    def get_sample(self, sample_index: int):
        return SHDSample(
            times=self.times_arr[sample_index],
            units=self.units_arr[sample_index],
            label_index=int(self.label_index_arr[sample_index]),
            speaker_index=int(self.speaker_index_arr[sample_index]),
            audio_filename=str(self.audio_filenames[sample_index]),
        )

    ### Audio stuff
    def get_audio_path(self, filename: str):
        return self.path / "hd_audio/audio" / filename
