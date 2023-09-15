"""Downloads 3D objects from all Objaverse-XL sources."""

from typing import Callable, Dict, Optional

import pandas as pd

from objaverse_xl.abstract import ObjaverseSource
from objaverse_xl.github import GitHubDownloader
from objaverse_xl.objaverse_v1 import SketchfabDownloader
from objaverse_xl.smithsonian import SmithsonianDownloader
from objaverse_xl.thingiverse import ThingiverseDownloader


class ObjaverseXLDownloader(ObjaverseSource):
    """Downloads 3D objects from all Objaverse-XL sources."""

    def __init__(self):
        super().__init__()

        self.downloaders = {
            "github": GitHubDownloader(),
            "thingiverse": ThingiverseDownloader(),
            "smithsonian": SmithsonianDownloader(),
            "sketchfab": SketchfabDownloader(),
        }

    def get_annotations(self, download_dir: str = "~/.objaverse") -> pd.DataFrame:
        """Loads the 3D object metadata as a Pandas DataFrame.

        Args:
            download_dir (str, optional): Directory to download the parquet metadata
                file. Supports all file systems supported by fsspec. Defaults to
                "~/.objaverse".

        Returns:
            pd.DataFrame: Metadata of the 3D objects as a Pandas DataFrame with columns
                for the object "fileIdentifier", "license", "source", "fileType",
                "sha256", and "metadata".
        """
        annotations = [
            downloader.get_annotations(download_dir)
            for downloader in self.downloaders.values()
        ]
        return pd.concat(annotations, ignore_index=True)

    def download_objects(
        self,
        objects: pd.DataFrame,
        download_dir: str = "~/.objaverse",
        processes: Optional[int] = None,
        handle_found_object: Optional[Callable] = None,
        handle_modified_object: Optional[Callable] = None,
        handle_missing_object: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Downloads all objects from the source.

        Args:
            objects (pd.DataFrame): Objects to download. Must have columns for
                the object "fileIdentifier" and "sha256". Use the `get_annotations`
                function to get the metadata.
            download_dir (str, optional): Directory to download the objects to.
                Supports all file systems supported by fsspec. Defaults to
                "~/.objaverse".
            processes (Optional[int], optional): Number of processes to use for
                downloading.  If None, will use the number of CPUs on the machine.
                Defaults to None.
            save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]],
                optional): Format to save the repository. If None, the repository will
                not be saved. If "files" is specified, each file will be saved
                individually. Otherwise, the repository can be saved as a "zip", "tar",
                or "tar.gz" file. Defaults to None.
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
            Dict[str, str]: Mapping of file identifiers to local paths of the downloaded
                3D objects.
        """
        sources = set(objects["source"].unique().tolist())
        all_sources = {"github", "thingiverse", "smithsonian", "sketchfab"}

        if not sources.issubset(all_sources):
            raise ValueError(
                f"Invalid sources: {sources}. Must be a subset of {all_sources}."
            )

        downloaded_objects = {}
        for source in sources:
            source_downloads = self.downloaders[source].download_objects(
                objects[objects["source"] == source],
                download_dir,
                processes,
                handle_found_object,
                handle_modified_object,
                handle_missing_object,
                **kwargs,
            )
            downloaded_objects.update(source_downloads)

        return downloaded_objects
