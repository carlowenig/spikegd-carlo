import gzip
import shutil
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import requests
from torch.utils.data import IterableDataset
from tqdm import tqdm

T = TypeVar("T")


# Inspired by torch.datasets.MNIST
class OnlineDataset(ABC, IterableDataset[T]):
    mirrors: list[str]
    """Available mirrors to download the dataset from. Should end with a slash."""

    mirror: str | None
    """
    The mirror which was used to download the dataset.
    """

    def __init__(
        self,
        data_root: str | Path,
        download: bool = True,
        name: str | None = None,
        verbose=False,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.name = name or type(self).__name__
        self.path = self.data_root / self.name
        self.verbose = verbose
        self.mirror = None

        if download:
            self.download()

        try:
            self._expect_exists()
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Dataset {self.name} does not exist at expected path {self.path}. "
                "Set download=True to download the dataset."
            ) from e

    @abstractmethod
    def _download_from_current_mirror(self):
        """Try to download the dataset from the current mirror (`self.mirror`)."""
        raise NotImplementedError

    @abstractmethod
    def _expect_exists(self):
        """Check if the dataset exists at the expected path.
        Raise an exception if not."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def _download_file(
        self,
        url_path: str,
        folder_path: str | Path | None = None,
        filename: str | None = None,
    ):
        """Helper function to download a file from a URL."""

        folder_path = self.get_path(folder_path)
        if filename is None:
            filename = url_path.split("/")[-1]
        file_path = folder_path / filename

        url = f"{self.mirror}{url_path}"

        self._verbose_print(f"  -> Downloading file {url} to {folder_path}")

        with requests.get(url, stream=True) as res:
            res.raise_for_status()
            total = int(res.headers.get("content-length", 0))

            with (
                open(file_path, "wb") as file,
                tqdm(
                    disable=not self.verbose,
                    desc="     ",
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                ) as bar,
            ):
                for data in res.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

        # download_url(f"{self.mirror}{url_path}", self.get_path(dest))

    def _extract_gzip(
        self,
        archive_path: str | Path,
        file_path: str | Path | None = None,
        remove_archive: bool = True,
    ):
        """Helper function to extract a gzip archive."""

        archive_path = self.get_path(archive_path)
        file_path = (
            archive_path.with_suffix("")
            if file_path is None
            else self.get_path(file_path)
        )

        self._verbose_print(
            f"  -> Extracting gzip archive {archive_path} to {file_path}"
        )

        with gzip.open(archive_path, "rb") as f_in:
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if remove_archive:
            archive_path.unlink()

    def _extract_tar(
        self,
        archive_path: str | Path,
        folder_path: str | Path | None = None,
        remove_archive: bool = True,
    ):
        """Helper function to extract a tar archive."""

        archive_path = self.get_path(archive_path)
        folder_path = self.get_path(folder_path)

        self._verbose_print(
            f"  -> Extracting tar archive {archive_path} to {folder_path}"
        )

        with (
            tarfile.open(archive_path, "r") as tar,
            tqdm(
                disable=not self.verbose,
                desc="     ",
                unit="B",
                unit_scale=True,
                miniters=1,
            ) as pbar,
        ):
            members = tar.getmembers()
            pbar.total = sum(member.size for member in members)

            for member in members:
                pbar.set_postfix_str(
                    f"{member.name:20.20} ({tqdm.format_sizeof(member.size, "B")})"
                )
                tar.extract(member, folder_path)
                pbar.update(member.size)

        if remove_archive:
            self._verbose_print("  -> Removing archive")
            archive_path.unlink()

    def _expect_file(self, path: str | Path | None):
        path = self.get_path(path)
        if not path.exists():
            raise FileNotFoundError(f"Expected file {path} does not exist")
        if not path.is_file():
            raise FileNotFoundError(f"Expected file {path} is not a file")

    def _expect_dir(self, path: str | Path | None):
        path = self.get_path(path)
        if not path.exists():
            raise FileNotFoundError(f"Expected directory {path} does not exist")
        if not path.is_dir():
            raise FileNotFoundError(f"Expected directory {path} is not a directory")

    def exists(self):
        try:
            self._expect_exists()
            return True
        except FileNotFoundError:
            return False

    def get_path(self, path: str | Path | None):
        if path is None:
            return self.path

        return self.path / path

    def download(self):
        """Download the data if it doesn't exist already."""

        if self.exists():
            return

        if self.path.exists():
            print(f"Removing incomplete dataset {self.name} at {self.path}...")
            shutil.rmtree(self.path)

        self.path.mkdir(parents=True)

        self._verbose_print(f"Downloading dataset {self.name} to {self.path}...")

        exceptions = []

        for i, mirror in enumerate(self.mirrors):
            self.mirror = mirror

            try:
                self._verbose_print(f"- Using mirror #{i+1}: '{mirror}'")

                self._download_from_current_mirror()
            except Exception as e:
                self._verbose_print(f"  -> Failed: {str(e).replace('\n', '\n     ')}")
                exceptions.append(e)
                continue
            finally:
                print()
            break
        else:
            raise ExceptionGroup(
                f"Could not download dataset {self.name}. "
                "None of the mirrors worked.",
                exceptions,
            )

        try:
            self._expect_exists()
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Dataset {self.name} does not exist at expected"
                f"path {self.path} after download"
            ) from e

    def _verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
