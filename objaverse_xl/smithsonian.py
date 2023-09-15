"""Script to download 3D objects from the Smithsonian Institution."""

import multiprocessing
import os
import tempfile
from multiprocessing import Pool
from typing import Callable, Dict, Optional, Tuple

import fsspec
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from objaverse_xl.abstract import ObjaverseSource
from objaverse_xl.utils import get_file_hash, get_uid_from_str


class SmithsonianDownloader(ObjaverseSource):
    """Script to download objects from the Smithsonian Institute."""

    def get_annotations(self, download_dir: str = "~/.objaverse") -> pd.DataFrame:
        """Loads the Smithsonian Object Metadata dataset as a Pandas DataFrame.

        Args:
            download_dir (str, optional): Directory to download the parquet metadata file.
                Supports all file systems supported by fsspec. Defaults to "~/.objaverse".

        Returns:
            pd.DataFrame: Smithsonian Object Metadata dataset as a Pandas DataFrame with
                columns for the object "title", "url", "quality", "file_type", "uid", and
                "license". The quality is always Medium and the file_type is always glb.
        """
        filename = os.path.join(download_dir, "smithsonian", "object-metadata.parquet")
        fs, path = fsspec.core.url_to_fs(filename)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        # download the parquet file if it doesn't exist
        if not fs.exists(path):
            url = "https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/smithsonian/object-metadata.parquet"
            response = requests.get(url)
            response.raise_for_status()
            with fs.open(path, "wb") as file:
                file.write(response.content)

        # load the parquet file with fsspec
        with fs.open(path) as f:
            df = pd.read_parquet(f)

        return df

    def _download_smithsonian_object(
        self,
        file_identifier: str,
        download_dir: Optional[str],
        expected_sha256: str,
        handle_found_object: Optional[Callable],
        handle_modified_object: Optional[Callable],
        handle_missing_object: Optional[Callable],
    ) -> Tuple[str, Optional[str]]:
        """Downloads a Smithsonian Object from a URL.

        Overwrites the file if it already exists and assumes this was previous checked.

        Args:
            file_identifier (str): URL to download the Smithsonian Object from.
            download_dir (Optional[str]): Directory to download the Smithsonian Object
                to. Supports all file systems supported by fsspec. If None, the
                Smithsonian Object will be deleted after it is downloaded and processed
                with the handler functions.
            expected_sha256 (str): The expected SHA256 of the contents of the downloaded
                object.
            handle_found_object (Optional[Callable]): Called when an object is
                successfully found and downloaded. Here, the object has the same sha256
                as the one that was downloaded with Objaverse-XL. If None, the object
                will be downloaded, but nothing will be done with it. Args for the
                function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): File identifier of the 3D object.
                - sha256 (str): SHA256 of the contents of the 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.
            handle_modified_object (Optional[Callable]): Called when a modified object
                is found and downloaded. Here, the object is successfully downloaded,
                but it has a different sha256 than the one that was downloaded with
                Objaverse-XL. This is not expected to happen very often, because the
                same commit hash is used for each repo. If None, the object will be
                downloaded, but nothing will be done with it. Args for the function
                include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): File identifier of the 3D object.
                - new_sha256 (str): SHA256 of the contents of the newly downloaded 3D
                    object.
                - old_sha256 (str): Expected SHA256 of the contents of the 3D object as
                    it was when it was downloaded with Objaverse-XL.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.
            handle_missing_object (Optional[Callable]): Called when an object that is in
                Objaverse-XL is not found. Here, it is likely that the repository was
                deleted or renamed. If None, nothing will be done with the missing
                object. Args for the function include:
                - file_identifier (str): File identifier of the 3D object.
                - sha256 (str): SHA256 of the contents of the original 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.


        Returns:
            Tuple[str, Optional[str]]: Tuple of the URL and the path to the downloaded
                Smithsonian Object. If the Smithsonian Object was not downloaded, the path
                will be None.
        """
        uid = get_uid_from_str(file_identifier)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, f"{uid}.glb")
            temp_path_tmp = f"{temp_path}.tmp"

            response = requests.get(file_identifier)

            # check if the path is valid
            if response.status_code == 404:
                logger.warning(f"404 for {file_identifier}")
                if handle_missing_object is not None:
                    handle_missing_object(
                        file_identifier=file_identifier,
                        sha256=expected_sha256,
                        metadata={},
                    )
                return file_identifier, None

            with open(temp_path_tmp, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            # rename to temp_path
            os.rename(temp_path_tmp, temp_path)

            # check the sha256
            sha256 = get_file_hash(temp_path)

            if sha256 == expected_sha256:
                if handle_found_object is not None:
                    handle_found_object(
                        local_path=temp_path,
                        file_identifier=file_identifier,
                        sha256=sha256,
                        metadata={},
                    )
            else:
                if handle_modified_object is not None:
                    handle_modified_object(
                        local_path=temp_path,
                        file_identifier=file_identifier,
                        new_sha256=sha256,
                        old_sha256=expected_sha256,
                        metadata={},
                    )

            if download_dir is not None:
                filename = os.path.join(
                    download_dir, "smithsonian", "objects", f"{uid}.glb"
                )
                fs, path = fsspec.core.url_to_fs(filename)
                fs.makedirs(os.path.dirname(path), exist_ok=True)
                fs.put(temp_path, path)
            else:
                path = None

        return file_identifier, path

    def _parallel_download_object(self, args):
        # workaround since starmap doesn't work well with tqdm
        return self._download_smithsonian_object(*args)

    def download_objects(
        self,
        objects: pd.DataFrame,
        download_dir: Optional[str] = "~/.objaverse",
        processes: Optional[int] = None,
        handle_found_object: Optional[Callable] = None,
        handle_modified_object: Optional[Callable] = None,
        handle_missing_object: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Downloads all Smithsonian Objects.

        Args:
            objects (pd.DataFrmae): Objects to download. Must have columns for
                the object "fileIdentifier" and "sha256". Use the `get_annotations`
                function to get the metadata.
            download_dir (Optional[str], optional): Directory to download the
                Smithsonian Objects to. Supports all file systems supported by fsspec.
                If None, the Smithsonian Objects will be deleted after they are
                downloaded and processed with the handler functions. Defaults to
                "~/.objaverse".
            processes (Optional[int], optional): Number of processes to use for
                downloading the Smithsonian Objects. If None, the number of processes
                will be set to the number of CPUs on the machine
                (multiprocessing.cpu_count()). Defaults to None.
            handle_found_object (Optional[Callable], optional): Called when an object is
                successfully found and downloaded. Here, the object has the same sha256
                as the one that was downloaded with Objaverse-XL. If None, the object
                will be downloaded, but nothing will be done with it. Args for the
                function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): File identifier of the 3D object.
                - sha256 (str): SHA256 of the contents of the 3D object.
                - metadata (Dict[Hashable, Any]): Metadata about the 3D object,
                    including the GitHub organization and repo names.
                Return is not used. Defaults to None.
            handle_modified_object (Optional[Callable], optional): Called when a
                modified object is found and downloaded. Here, the object is
                successfully downloaded, but it has a different sha256 than the one that
                was downloaded with Objaverse-XL. This is not expected to happen very
                often, because the same commit hash is used for each repo. If None, the
                object will be downloaded, but nothing will be done with it. Args for
                the function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): File identifier of the 3D object.
                - new_sha256 (str): SHA256 of the contents of the newly downloaded 3D
                    object.
                - old_sha256 (str): Expected SHA256 of the contents of the 3D object as
                    it was when it was downloaded with Objaverse-XL.
                - metadata (Dict[Hashable, Any]): Metadata about the 3D object, which is
                    particular to the souce.
                Return is not used. Defaults to None.
            handle_missing_object (Optional[Callable], optional): Called when an object
                that is in Objaverse-XL is not found. Here, it is likely that the
                repository was deleted or renamed. If None, nothing will be done with
                the missing object.
                Args for the function include:
                - file_identifier (str): File identifier of the 3D object.
                - sha256 (str): SHA256 of the contents of the original 3D object.
                - metadata (Dict[Hashable, Any]): Metadata about the 3D object, which is
                    particular to the source.
                Return is not used. Defaults to None.

        Returns:
            Dict[str, str]: A dictionary mapping from the fileIdentifier to the
                download_path.
        """
        if processes is None:
            processes = multiprocessing.cpu_count()

        out = {}
        objects_to_download = []
        if download_dir is not None:
            objects_dir = os.path.join(download_dir, "smithsonian", "objects")
            fs, path = fsspec.core.url_to_fs(objects_dir)
            fs.makedirs(path, exist_ok=True)

            # get the existing glb files
            existing_glb_files = fs.glob(
                os.path.join(objects_dir, "*.glb"), refresh=True
            )
            existing_uids = set(
                os.path.basename(file).split(".")[0] for file in existing_glb_files
            )

            # find the urls that need to be downloaded
            already_downloaded_objects = set()
            for _, item in objects.iterrows():
                file_identifier = item["fileIdentifier"]
                uid = get_uid_from_str(file_identifier)
                if uid not in existing_uids:
                    objects_to_download.append(item)
                else:
                    already_downloaded_objects.add(file_identifier)
                out[file_identifier] = os.path.join(
                    os.path.expanduser(objects_dir), f"{uid}.glb"
                )
        else:
            existing_uids = set()
            objects_to_download = [item for _, item in objects.iterrows()]
            already_downloaded_objects = set()
            out = {}

        logger.info(
            f"Found {len(already_downloaded_objects)} Smithsonian Objects already downloaded"
        )
        logger.info(
            f"Downloading {len(objects_to_download)} Smithsonian Objects with {processes} processes"
        )

        if len(objects_to_download) == 0:
            return out

        args = [
            [
                item["fileIdentifier"],
                download_dir,
                item["sha256"],
                handle_found_object,
                handle_modified_object,
                handle_missing_object,
            ]
            for item in objects_to_download
        ]
        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self._parallel_download_object, args),
                    total=len(objects_to_download),
                    desc="Downloading Smithsonian Objects",
                )
            )

        for file_identifier, download_path in results:
            if download_path is not None:
                out[file_identifier] = download_path

        return out
