"""Script to download objects from Objaverse 1.0."""

import gzip
import json
import multiprocessing
import os
import tempfile
import urllib.request
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import fsspec
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from objaverse_xl.abstract import ObjaverseSource
from objaverse_xl.utils import get_file_hash


class SketchfabDownloader(ObjaverseSource):
    """A class for downloading and processing Objaverse 1.0."""

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
        remote_url = "https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/objaverse_v1/object-metadata.parquet"
        download_path = os.path.join(
            download_dir, "hf-objaverse-v1", "thingiverse-objects.parquet"
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

        annotations_df["metadata"] = "{}"

        return annotations_df

    def load_full_annotations(
        self,
        uids: Optional[List[str]] = None,
        download_dir: str = "~/.objaverse",
    ) -> Dict[str, Any]:
        """Load the full metadata of all objects in the dataset.

        Args:
            uids: A list of uids with which to load metadata. If None, it loads
                the metadata for all uids.
            download_dir: The base directory to download the annotations to. Supports all
                file systems supported by fsspec. Defaults to "~/.objaverse".

        Returns:
            A dictionary of the metadata for each object. The keys are the uids and the
            values are the metadata for that object.
        """
        # make the metadata dir if it doesn't exist
        metadata_path = os.path.join(download_dir, "hf-objaverse-v1", "metadata")
        fs, _ = fsspec.core.url_to_fs(metadata_path)
        fs.makedirs(metadata_path, exist_ok=True)

        # get the dir ids that need to be loaded if only downloading a subset of uids
        object_paths = self._load_object_paths(download_dir=download_dir)
        dir_ids = (
            {object_paths[uid].split("/")[1] for uid in uids}
            if uids is not None
            else {f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)}
        )

        # get the existing metadata files
        existing_metadata_files = fs.glob(
            os.path.join(metadata_path, "*.json.gz"), refresh=True
        )
        existing_dir_ids = {
            file.split("/")[-1].split(".")[0]
            for file in existing_metadata_files
            if file.endswith(".json.gz")  # note partial files end with .json.gz.tmp
        }
        downloaded_dir_ids = existing_dir_ids.intersection(dir_ids)
        logger.info(
            f"Found {len(downloaded_dir_ids)} metadata files already downloaded"
        )

        # download the metadata from the missing dir_ids
        dir_ids_to_download = dir_ids - existing_dir_ids
        logger.info(f"Downloading {len(dir_ids_to_download)} metadata files")

        # download the metadata file if it doesn't exist
        if len(dir_ids_to_download) > 0:
            for i_id in tqdm(dir_ids_to_download, desc="Downloading metadata files"):
                # get the path to the json file
                path = os.path.join(metadata_path, f"{i_id}.json.gz")

                # get the url to the remote json file
                hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"

                # download the file to a tmp path to avoid partial downloads on interruption
                tmp_path = f"{path}.tmp"
                with fs.open(tmp_path, "wb") as f:
                    with urllib.request.urlopen(hf_url) as response:
                        f.write(response.read())
                fs.rename(tmp_path, path)

        out = {}
        for i_id in tqdm(dir_ids, desc="Reading metadata files"):
            # get the path to the json file
            path = os.path.join(metadata_path, f"{i_id}.json.gz")

            # read the json file of the metadata chunk
            with fs.open(path, "rb") as f:
                with gzip.GzipFile(fileobj=f) as gfile:
                    content = gfile.read()
                    data = json.loads(content)

            # filter the data to only include the uids we want
            if uids is not None:
                data = {uid: data[uid] for uid in uids if uid in data}

            # add the data to the out dict
            out.update(data)

        return out

    def _load_object_paths(self, download_dir: str) -> Dict[str, str]:
        """Load the object paths from the dataset.

        The object paths specify the location of where the object is located in the
        Hugging Face repo.

        Returns:
            A dictionary mapping the uid to the object path.
        """
        object_paths_file = "object-paths.json.gz"
        local_path = os.path.join(download_dir, "hf-objaverse-v1", object_paths_file)

        # download the object_paths file if it doesn't exist
        fs, path = fsspec.core.url_to_fs(local_path)
        if not fs.exists(path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
            fs.makedirs(os.path.dirname(path), exist_ok=True)

            # download the file to a tmp path to avoid partial downloads on interruption
            tmp_path = f"{path}.tmp"
            with fs.open(tmp_path, "wb") as f:
                with urllib.request.urlopen(hf_url) as response:
                    f.write(response.read())
            fs.rename(tmp_path, path)

        # read the object_paths
        with fs.open(path, "rb") as f:
            with gzip.GzipFile(fileobj=f) as gfile:
                content = gfile.read()
                object_paths = json.loads(content)

        return object_paths

    def load_uids(self, download_dir: str = "~/.objaverse") -> List[str]:
        """Load the uids from the dataset.

        Returns:
            A list of all the UIDs from the dataset.
        """
        return list(self._load_object_paths(download_dir=download_dir).keys())

    def _download_object(
        self,
        file_identifier: str,
        hf_object_path: str,
        download_dir: Optional[str],
        expected_sha256: str,
        handle_found_object: Optional[Callable] = None,
        handle_modified_object: Optional[Callable] = None,
    ) -> Tuple[str, Optional[str]]:
        """Download the object for the given uid.

        Args:
            file_identifier: The file identifier of the object.
            hf_object_path: The path to the object in the Hugging Face repo. Here,
                hf_object_path is the part that comes after "main" in the Hugging Face
                repo url:
                https://huggingface.co/datasets/allenai/objaverse/resolve/main/{hf_object_path}
            download_dir: The base directory to download the object to. Supports all
                file systems supported by fsspec. Defaults to "~/.objaverse".
            expected_sha256 (str): The expected SHA256 of the contents of the downloade
                object.
            handle_found_object (Optional[Callable]): Called when an object is
                successfully found and downloaded. Here, the object has the same sha256
                as the one that was downloaded with Objaverse-XL. If None, the object
                will be downloaded, but nothing will be done with it. Args for the
                function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
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
                - file_identifier (str): GitHub URL of the 3D object.
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
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the original 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.


        Returns:
            A tuple of the uid and the path to where the downloaded object. If
            download_dir is None, the path will be None.
        """
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{hf_object_path}"

        with tempfile.TemporaryDirectory() as temp_dir:
            # download the file locally
            temp_path = os.path.join(temp_dir, hf_object_path)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            temp_path_tmp = f"{temp_path}.tmp"
            with open(temp_path_tmp, "wb") as file:
                with urllib.request.urlopen(hf_url) as response:
                    file.write(response.read())
            os.rename(temp_path_tmp, temp_path)

            # get the sha256 of the downloaded file
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
                filename = os.path.join(download_dir, "hf-objaverse-v1", hf_object_path)
                fs, path = fsspec.core.url_to_fs(filename)
                fs.makedirs(os.path.dirname(path), exist_ok=True)
                fs.put(temp_path, path)
            else:
                path = None

        return file_identifier, path

    def _parallel_download_object(self, args):
        # workaround since starmap doesn't work well with tqdm
        return self._download_object(*args)

    def _get_uid(self, item: pd.Series) -> str:
        file_identifier = item["fileIdentifier"]
        return file_identifier.split("/")[-1]

    def uid_to_file_identifier(self, uid: str) -> str:
        """Convert the uid to the file identifier.

        Args:
            uid (str): The uid of the object.

        Returns:
            The file identifier of the object.
        """
        return f"https://sketchfab.com/3d-models/{uid}"

    def file_identifier_to_uid(self, file_identifier: str) -> str:
        """Convert the file identifier to the uid.

        Args:
            file_identifier (str): The file identifier of the object.

        Returns:
            The uid of the object.
        """
        return file_identifier.split("/")[-1]

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
        """Return the path to the object files for the given uids.

        If the object is not already downloaded, it will be downloaded.

        Args:
            objects (pd.DataFrame): Objects to download. Must have columns for
                the object "fileIdentifier" and "sha256". Use the `get_annotations`
                function to get the metadata.
            download_dir (Optional[str], optional): The base directory to download the
                object to. Supports all file systems supported by fsspec. If None, the
                objects will be removed after downloading. Defaults to "~/.objaverse".
            processes (Optional[int], optional): The number of processes to use to
                download the objects. If None, the number of processes will be set to
                the number of CPUs on the machine (multiprocessing.cpu_count()).
                Defaults to None.
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
            A dictionary mapping the object fileIdentifier to the local path of where
            the object downloaded.
        """
        hf_object_paths = self._load_object_paths(
            download_dir=download_dir if download_dir is not None else "~/.objaverse"
        )
        if processes is None:
            processes = multiprocessing.cpu_count()

        # make a copy of the objects so we don't modify the original
        objects = objects.copy()
        objects["uid"] = objects.apply(self._get_uid, axis=1)
        uids_to_sha256 = dict(zip(objects["uid"], objects["sha256"]))
        uids_set = set(uids_to_sha256.keys())

        # create a new df where the uids are the index
        objects_uid_index = objects.set_index("uid")

        out = {}
        objects_to_download = []
        if download_dir is None:
            for _, item in objects.iterrows():
                uid = item["uid"]
                if uid not in hf_object_paths:
                    logger.error(f"Could not find object with uid {uid}!")
                    if handle_missing_object is not None:
                        handle_missing_object(
                            file_identifier=item["fileIdentifier"],
                            sha256=item["sha256"],
                            metadata={},
                        )
                    continue
                objects_to_download.append(
                    (item["fileIdentifier"], hf_object_paths[uid], item["sha256"])
                )
        else:
            versioned_dirname = os.path.join(download_dir, "hf-objaverse-v1")
            fs, path = fsspec.core.url_to_fs(versioned_dirname)

            # Get the existing file paths. This is much faster than calling fs.exists() for each
            # file. `glob()` is like walk, but returns a list of files instead of the nested
            # directory structure. glob() is also faster than find() / walk() since it doesn't
            # need to traverse the entire directory structure.
            existing_file_paths = fs.glob(
                os.path.join(path, "glbs", "*", "*.glb"), refresh=True
            )
            existing_uids = {
                file.split("/")[-1].split(".")[0]
                for file in existing_file_paths
                if file.endswith(".glb")  # note partial files end with .glb.tmp
            }

            # add the existing downloaded uids to the return dict
            already_downloaded_uids = uids_set.intersection(existing_uids)
            for uid in already_downloaded_uids:
                hf_object_path = hf_object_paths[uid]
                fs_abs_object_path = os.path.join(versioned_dirname, hf_object_path)
                out[self.uid_to_file_identifier(uid)] = fs_abs_object_path

            logger.info(
                f"Found {len(already_downloaded_uids)} objects already downloaded"
            )

            # get the uids that need to be downloaded
            remaining_uids = uids_set - existing_uids
            for uid in remaining_uids:
                item = objects_uid_index.loc[uid]
                if uid not in hf_object_paths:
                    logger.error(f"Could not find object with uid {uid}. Skipping it.")
                    if handle_missing_object is not None:
                        handle_missing_object(
                            file_identifier=item["fileIdentifier"],
                            sha256=item["sha256"],
                            metadata={},
                        )
                    continue
                objects_to_download.append(
                    (item["fileIdentifier"], hf_object_paths[uid], item["sha256"])
                )

            logger.info(
                f"Downloading {len(objects_to_download)} new objects across {processes} processes"
            )

        # check if all objects are already downloaded
        if len(objects_to_download) == 0:
            return out

        args = [
            (
                file_identifier,
                hf_object_path,
                download_dir,
                sha256,
                handle_found_object,
                handle_modified_object,
            )
            for file_identifier, hf_object_path, sha256 in objects_to_download
        ]

        # download the objects in parallel
        with Pool(processes) as pool:
            new_object_downloads = list(
                tqdm(
                    pool.imap_unordered(self._parallel_download_object, args),
                    total=len(args),
                )
            )

        for file_identifier, local_path in new_object_downloads:
            out[file_identifier] = local_path

        return out

    def load_lvis_annotations(
        self,
        download_dir: str = "~/.objaverse",
    ) -> Dict[str, List[str]]:
        """Load the LVIS annotations.

        If the annotations are not already downloaded, they will be downloaded.

        Args:
            download_dir: The base directory to download the annotations to. Supports all
            file systems supported by fsspec. Defaults to "~/.objaverse".

        Returns:
            A dictionary mapping the LVIS category to the list of uids in that category.
        """
        hf_url = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"

        download_path = os.path.join(
            download_dir, "hf-objaverse-v1", "lvis-annotations.json.gz"
        )

        # use fsspec
        fs, path = fsspec.core.url_to_fs(download_path)
        if not fs.exists(path):
            # make dir if it doesn't exist
            fs.makedirs(os.path.dirname(path), exist_ok=True)

            # download the file
            with fs.open(path, "wb") as f:
                with urllib.request.urlopen(hf_url) as response:
                    f.write(response.read())

        # load the gzip file
        with fs.open(path, "rb") as f:
            with gzip.GzipFile(fileobj=f) as gfile:
                content = gfile.read()
                data = json.loads(content)

        return data
