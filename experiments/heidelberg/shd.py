from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision.datasets.utils import download_url, extract_archive


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


class Resource:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load_impl(self, root: str | Path):
        raise NotImplementedError

    def exists(self, root: str | Path) -> bool:
        return (Path(root) / self.path).exists()

    def load(self, root: str | Path, check=True):
        if self.exists(root):
            return

        (Path(root) / self.path).parent.mkdir(parents=True, exist_ok=True)

        self._load_impl(root)

        if check and not self.exists(root):
            raise RuntimeError(f"Failed to load {self.path}")

    def full_path(self, root: str | Path) -> Path:
        return Path(root) / self.path


class OnlineFileResource(Resource):
    def __init__(
        self, url: str, filename: str | None = None, base_urls: tuple[str, ...] = ()
    ):
        self.url = url
        self.filename = filename or Path(url).name
        self.base_urls: tuple[str, ...] = base_urls or ("",)
        super().__init__(self.filename)

    def _load_impl(self, root: str | Path):
        for base_url in self.base_urls:
            full_url = f"{base_url}{self.url}"
            try:
                print(f"Downloading {full_url} to {root}")

                download_url(full_url, root, self.filename)
            except URLError as error:
                print(f"Failed to download (trying next):\n{error}")
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError(f"Error downloading {self.url}")


class ExtractedFolderResource(Resource):
    def __init__(
        self, archive: Resource, path: str | Path, remove_archive: bool = True
    ):
        self.archive = archive
        self.remove_archive = remove_archive
        super().__init__(path)

    def _load_impl(self, root: str | Path):
        self.archive.load(root)
        extract_archive(
            self.archive.full_path(root),
            self.full_path(root),
            remove_finished=self.remove_archive,
        )


class ExtractedFileResource(Resource):
    def __init__(self, archive: Resource, filename: str, remove_archive: bool = True):
        self.archive = archive
        self.remove_archive = remove_archive
        super().__init__(filename)

    def _load_impl(self, root: str | Path):
        self.archive.load(root)
        extract_archive(
            self.archive.full_path(root), root, remove_finished=self.remove_archive
        )


class NestedResource(Resource):
    def __init__(self, resource: Resource, folder: str | Path):
        self.resource = resource
        self.folder = Path(folder)
        super().__init__(self.folder / self.resource.path)

    def _load_impl(self, root: str | Path):
        return self.resource._load_impl(root / self.folder)


# Inspired by torch.datasets.MNIST
class SHD(IterableDataset[SHDSample]):
    """Spiking Heidelberg digits (SHD) dataset."""

    mirrors = (
        "https://zenkelab.org/datasets/",
        # TODO: Maybe add more mirrors
    )

    def __init__(
        self,
        root: str | Path,
        train=True,
        download=True,
    ):
        super().__init__()

        self.root = Path(root)
        self.train = train
        self._data_filename = f"shd_{'train' if train else 'test'}.h5"

        self.resources: list[Resource] = [
            OnlineFileResource("README.md", base_urls=self.mirrors),
            ExtractedFileResource(
                archive=OnlineFileResource(
                    f"{self._data_filename}.gz", base_urls=self.mirrors
                ),
                filename=self._data_filename,
            ),
            ExtractedFolderResource(
                archive=OnlineFileResource("hd_audio.tar.gz", base_urls=self.mirrors),
                path="hd_audio",
            ),
        ]

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        ### Data
        self.data_file = h5py.File(self.raw_folder / self._data_filename)

        # times_arr and units_arr are arrays of numpy arrays
        self.times_arr = self._get_dataset_as_arr("spikes/times")
        self.units_arr = self._get_dataset_as_arr("spikes/units")
        self.label_index_arr = self._get_dataset_as_arr("labels")
        self.speaker_index_arr = self._get_dataset_as_arr("extra/speaker")

        self.Nsamples = len(self.times_arr)

        assert (
            self.units_arr.shape
            == self.times_arr.shape
            == self.label_index_arr.shape
            == self.speaker_index_arr.shape
            == (self.Nsamples,)
        )

        # Store label names as array for easy access
        self.label_names = self._get_dataset_as_arr("extra/keys", str)

        # Store speaker info as arrays for easy access
        self.ages_by_speaker = self._get_dataset_as_arr("extra/meta_info/age")
        self.genders_by_speaker = self._get_dataset_as_arr(
            "extra/meta_info/gender", str
        )
        self.body_heights_by_speaker = self._get_dataset_as_arr(
            "extra/meta_info/body_height"
        )

        self.age_arr = self.ages_by_speaker[self.speaker_index_arr]
        self.gender_arr = self.genders_by_speaker[self.speaker_index_arr]
        self.body_height_arr = self.body_heights_by_speaker[self.speaker_index_arr]

        # Derived quantities
        self.spike_count_arr = np.array([len(times) for times in self.times_arr])
        self.trial_length_arr = np.array([times[-1] for times in self.times_arr])
        self.spike_rate_arr = self.spike_count_arr / self.trial_length_arr

        ### Audio
        self.audio_filenames = np.loadtxt(
            self.raw_folder
            / "hd_audio"
            / f"{"train" if train else "test"}_filenames.txt",
            dtype=str,
            delimiter=None,
        )

    @property
    def raw_folder(self):
        return self.root / self.__class__.__name__ / "raw"

    def _check_exists(self):
        return all(resource.exists(self.raw_folder) for resource in self.resources)

    def download(self) -> None:
        """Download the Heidelberg data if it doesn't exist already."""

        if self._check_exists():
            return

        for resource in self.resources:
            resource.load(self.raw_folder)

    def _get_dataset_as_arr(self, path: str, dtype=None):
        ds = self.data_file[path]
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

    # def _get_dataset_arr(self, path: str, dtype=None):
    #     ds = self._get_dataset(path)
    #     arr = ds[:]
    #     assert isinstance(arr, np.ndarray)

    #     if dtype is not None:
    #         arr = arr.astype(dtype)

    #     return arr

    def __len__(self):
        return len(self._get_dataset_as_arr("spikes/times"))

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

    def __getitem__(self, sample_index):
        return self.get_sample(sample_index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    ### Audio stuff
    def get_audio_path(self, filename: str):
        return self.raw_folder / "hd_audio" / "audio" / filename
