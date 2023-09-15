"""Script to download objects from Thingiverse."""

import multiprocessing
import os
import tempfile
import time
from multiprocessing import Pool
from typing import Callable, Dict, Optional, Tuple

import fsspec
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from objaverse_xl.abstract import ObjaverseSource
from objaverse_xl.utils import get_file_hash


class ThingiverseDownloader(ObjaverseSource):
    """Script to download objects from Thingiverse."""

    def get_annotations(self, download_dir: str = "~/.objaverse") -> pd.DataFrame:
        """Load the annotations from the given directory.

        Args:
            download_dir (str, optional): The directory to load the annotations from.
                Supports all file systems supported by fsspec. Defaults to
                "~/.objaverse".

        Returns:
            pd.DataFrame: The annotations, which includes the columns "thingId", "fileId",
                "filename", and "license".
        """
        remote_url = "https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/thingiverse/thingiverse-objects.parquet"
        download_path = os.path.join(
            download_dir, "thingiverse", "thingiverse-objects.parquet"
        )
        fs, path = fsspec.core.url_to_fs(download_path)

        if not fs.exists(path):
            fs.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Downloading {remote_url} to {download_path}")
            response = requests.get(remote_url)
            response.raise_for_status()
            with fs.open(path, "wb") as file:
                file.write(response.content)

        # read the file with pandas and fsspec
        with fs.open(download_path, "rb") as f:
            annotations_df = pd.read_parquet(f)

        return annotations_df

    def _get_response_with_retries(
        self, url: str, max_retries: int = 3, retry_delay: int = 5
    ) -> Optional[requests.models.Response]:
        """Get a response from a URL with retries.

        Args:
            url (str): The URL to get a response from.
            max_retries (int, optional): The maximum number of retries. Defaults to 3.
            retry_delay (int, optional): The delay between retries in seconds. Defaults to 5.

        Returns:
            Optional[requests.models.Response]: The response from the URL. If there was an error, returns None.
        """

        for i in range(max_retries):
            try:
                response = requests.get(url, stream=True)
                # if successful, break out of loop
                if response.status_code not in {200, 404}:
                    time.sleep(retry_delay)
                    continue
                break
            except ConnectionError:
                if i < max_retries - 1:  # i.e. not on the last try
                    time.sleep(retry_delay)
        else:
            return None

        return response

    def _download_item(
        self,
        thingi_file_id: str,
        thingi_thing_id: str,
        file_identifier: str,
        download_dir: Optional[str],
        expected_sha256: str,
        handle_found_object: Optional[Callable],
        handle_modified_object: Optional[Callable],
        handle_missing_object: Optional[Callable],
    ) -> Tuple[str, Optional[str]]:
        """Download the given item.

        Args:
            thingi_file_id (str): The Thingiverse file ID of the object.
            thingi_thing_id (str): The Thingiverse thing ID of the object.
            file_identifier (str): File identifier of the Thingiverse object.
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
            Optional[str]: The path to the downloaded file. If there was an error or 404,
                returns None.
        """
        url = f"https://www.thingiverse.com/download:{thingi_file_id}"
        response = self._get_response_with_retries(url)
        filename = f"thing-{thingi_thing_id}-file-{thingi_file_id}.stl"

        if response is None:
            logger.warning(
                f"Thingiverse file ID {thingi_file_id} could not get response from {url}"
            )
            # NOTE: the object is probably not missing, but the request failed
            return file_identifier, None

        # Check if the request was successful
        if response.status_code == 404:
            logger.warning(
                f"Thingiverse file ID {thingi_file_id} (404) could not find file"
            )
            if handle_missing_object is not None:
                handle_missing_object(
                    file_identifier=file_identifier, sha256=expected_sha256, metadata={}
                )
            return file_identifier, None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, filename)
            temp_path_tmp = temp_path + ".tmp"

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
                filename = os.path.join(download_dir, filename)
                fs, path = fsspec.core.url_to_fs(filename)
                fs.makedirs(os.path.dirname(path), exist_ok=True)
                fs.put(temp_path, path)
            else:
                path = None

        return file_identifier, path

    def _parallel_download_item(self, args):
        return self._download_item(*args)

    def get_file_id_from_file_identifier(self, file_identifier: str) -> str:
        """Get the thingiverse file ID from the Objaverse-XL file identifier.

        Args:
            file_identifier (str): The Objaverse-XL file identifier.

        Returns:
            str: The Thingiverse file ID.
        """
        return file_identifier.split("fileId=")[-1]

    def get_thing_id_from_file_identifier(self, file_identifier: str) -> str:
        """Get the thingiverse thing ID from the Objaverse-XL file identifier.

        Args:
            file_identifier (str): The Objaverse-XL file identifier.

        Returns:
            str: The Thingiverse thing ID.
        """
        return file_identifier.split("/")[-2].split(":")[1]

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
        """Download the objects from the given list of things and files.

        Args:
            objects (pd.DataFrame): Thingiverse objects to download. Must have columns
                for the object "fileIdentifier" and "sha256". Use the `get_annotations`
                function to get the metadata.
            download_dir (str, optional): The directory to save the files to. Supports
                all file systems supported by fsspec. Defaults to "~/.objaverse-xl".
            processes (int, optional): The number of processes to use. If None, maps to
                use all available CPUs using multiprocessing.cpu_count(). Defaults to
                None.
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
            Dict[str, str]: A dictionary mapping from the fileIdentifier to the path of
                the downloaded file.
        """
        if processes is None:
            processes = multiprocessing.cpu_count()

        objects = objects.copy()
        objects["thingiFileId"] = objects["fileIdentifier"].apply(
            self.get_file_id_from_file_identifier
        )
        objects["thingiThingId"] = objects["fileIdentifier"].apply(
            self.get_thing_id_from_file_identifier
        )

        # create the download directory
        out = {}
        if download_dir is not None:
            download_dir = os.path.join(download_dir, "thingiverse")
            fs, path = fsspec.core.url_to_fs(download_dir)
            fs.makedirs(path, exist_ok=True)

            # check to filter out files that already exist
            existing_files = fs.glob(os.path.join(download_dir, "*.stl"), refresh=True)
            existing_file_ids = {
                os.path.basename(file).split(".")[0].split("-")[-1]
                for file in existing_files
            }

            # filter out existing files
            items_to_download = []
            already_downloaded_count = 0
            for _, item in objects.iterrows():
                if item["thingiFileId"] in existing_file_ids:
                    already_downloaded_count += 1
                    out[item["fileIdentifier"]] = os.path.join(
                        os.path.expanduser(download_dir),
                        f"thing-{item['thingiThingId']}-file-{item['thingiFileId']}.stl",
                    )
                else:
                    items_to_download.append(item)

            logger.info(
                f"Found {already_downloaded_count} Thingiverse objects downloaded"
            )
        else:
            items_to_download = [item for _, item in objects.iterrows()]

        logger.info(
            f"Downloading {len(items_to_download)} Thingiverse objects with {processes=}"
        )
        if len(items_to_download) == 0:
            return out

        # download the files
        args = [
            (
                item["thingiFileId"],
                item["thingiThingId"],
                item["fileIdentifier"],
                download_dir,
                item["sha256"],
                handle_found_object,
                handle_modified_object,
                handle_missing_object,
            )
            for item in items_to_download
        ]

        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self._parallel_download_item, args),
                    total=len(args),
                    desc="Downloading Thingiverse Objects",
                )
            )

        for file_identifier, download_path in results:
            if download_path is not None:
                out[file_identifier] = download_path

        return out
