import os
import shutil

import fsspec
import pandas as pd

from objaverse.xl.github import GitHubDownloader
from objaverse.xl.sketchfab import SketchfabDownloader
from objaverse.xl.thingiverse import ThingiverseDownloader
from objaverse.xl.smithsonian import SmithsonianDownloader


def test_github_process_repo():
    github_downloader = GitHubDownloader()
    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    for save_repo_format in ["tar", "tar.gz", "zip", "files"]:
        shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
        out = github_downloader._process_repo(
            repo_id="mattdeitke/objaverse-xl-test-files",
            fs=fs,
            base_dir=path,
            save_repo_format=save_repo_format,  # type: ignore
            expected_objects=dict(),
            handle_found_object=None,
            handle_modified_object=None,
            handle_missing_object=None,
            handle_new_object=handle_new_object,
            commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
        )

        # test that the sha256's are correct
        assert len(out) == 0
        sha256s = [x["sha256"] for x in new_objects]
        for sha256 in [
            "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
            "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9",
        ]:
            assert sha256 in sha256s, f"{sha256=} not in {sha256s=}"
        github_urls = [x["file_identifier"] for x in new_objects]
        for github_url in [
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj",
        ]:
            assert github_url in github_urls, f"{github_url=} not in {github_urls=}"

        # test that the files are correct
        if save_repo_format != "files":
            assert fs.exists(
                os.path.join(
                    path,
                    "repos",
                    "mattdeitke",
                    f"objaverse-xl-test-files.{save_repo_format}",
                )
            )
        else:
            assert fs.exists(
                os.path.join(
                    base_download_dir, "repos", "mattdeitke", "objaverse-xl-test-files"
                )
            )


def test_github_handle_new_object():
    github_downloader = GitHubDownloader()
    found_objects = []
    handle_found_object = (
        lambda local_path, file_identifier, sha256, metadata: found_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    missing_objects = []
    handle_missing_object = (
        lambda file_identifier, sha256, metadata: missing_objects.append(
            dict(
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    modified_objects = []
    handle_modified_object = lambda local_path, file_identifier, new_sha256, old_sha256, metadata: modified_objects.append(
        dict(
            local_path=local_path,
            file_identifier=file_identifier,
            new_sha256=new_sha256,
            old_sha256=old_sha256,
            metadata=metadata,
        )
    )

    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
    out = github_downloader._process_repo(
        repo_id="mattdeitke/objaverse-xl-test-files",
        fs=fs,
        base_dir=path,
        save_repo_format=None,
        expected_objects=dict(),
        handle_found_object=handle_found_object,
        handle_modified_object=handle_modified_object,
        handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
        commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
    )

    assert len(out) == 0
    assert len(new_objects) == 3
    assert len(found_objects) == 0
    assert len(modified_objects) == 0
    assert len(missing_objects) == 0


def test_github_handle_found_object():
    github_downloader = GitHubDownloader()
    found_objects = []
    handle_found_object = (
        lambda local_path, file_identifier, sha256, metadata: found_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    missing_objects = []
    handle_missing_object = (
        lambda file_identifier, sha256, metadata: missing_objects.append(
            dict(
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    modified_objects = []
    handle_modified_object = lambda local_path, file_identifier, new_sha256, old_sha256, metadata: modified_objects.append(
        dict(
            local_path=local_path,
            file_identifier=file_identifier,
            new_sha256=new_sha256,
            old_sha256=old_sha256,
            metadata=metadata,
        )
    )

    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
    out = github_downloader._process_repo(
        repo_id="mattdeitke/objaverse-xl-test-files",
        fs=fs,
        base_dir=path,
        save_repo_format=None,
        expected_objects={
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx": "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb": "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj": "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
        },
        handle_found_object=handle_found_object,
        handle_modified_object=handle_modified_object,
        handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
        commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
    )

    assert len(out) == 0
    assert len(found_objects) == 3
    assert len(missing_objects) == 0
    assert len(new_objects) == 0
    assert len(modified_objects) == 0


def test_github_handle_modified_object():
    github_downloader = GitHubDownloader()
    found_objects = []
    handle_found_object = (
        lambda local_path, file_identifier, sha256, metadata: found_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    missing_objects = []
    handle_missing_object = (
        lambda file_identifier, sha256, metadata: missing_objects.append(
            dict(
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    modified_objects = []
    handle_modified_object = lambda local_path, file_identifier, new_sha256, old_sha256, metadata: modified_objects.append(
        dict(
            local_path=local_path,
            file_identifier=file_identifier,
            new_sha256=new_sha256,
            old_sha256=old_sha256,
            metadata=metadata,
        )
    )

    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
    out = github_downloader._process_repo(
        repo_id="mattdeitke/objaverse-xl-test-files",
        fs=fs,
        base_dir=path,
        save_repo_format=None,
        expected_objects={
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx": "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9<modified>",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb": "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj": "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
        },
        handle_found_object=handle_found_object,
        handle_modified_object=handle_modified_object,
        handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
        commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
    )

    assert len(out) == 0
    assert len(found_objects) == 2
    assert len(missing_objects) == 0
    assert len(new_objects) == 0
    assert len(modified_objects) == 1


def test_github_handle_missing_object():
    github_downloader = GitHubDownloader()
    found_objects = []
    handle_found_object = (
        lambda local_path, file_identifier, sha256, metadata: found_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    missing_objects = []
    handle_missing_object = (
        lambda file_identifier, sha256, metadata: missing_objects.append(
            dict(
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    modified_objects = []
    handle_modified_object = lambda local_path, file_identifier, new_sha256, old_sha256, metadata: modified_objects.append(
        dict(
            local_path=local_path,
            file_identifier=file_identifier,
            new_sha256=new_sha256,
            old_sha256=old_sha256,
            metadata=metadata,
        )
    )

    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
    out = github_downloader._process_repo(
        repo_id="mattdeitke/objaverse-xl-test-files",
        fs=fs,
        base_dir=path,
        save_repo_format=None,
        expected_objects={
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx": "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example-2.fbx": "<fake-missing-object>",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb": "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj": "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
        },
        handle_found_object=handle_found_object,
        handle_modified_object=handle_modified_object,
        handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
        commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
    )

    assert len(out) == 0
    assert len(found_objects) == 3
    assert len(missing_objects) == 1
    assert len(new_objects) == 0
    assert len(modified_objects) == 0


def test_github_handle_missing_object_2():
    github_downloader = GitHubDownloader()
    found_objects = []
    handle_found_object = (
        lambda local_path, file_identifier, sha256, metadata: found_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    missing_objects = []
    handle_missing_object = (
        lambda file_identifier, sha256, metadata: missing_objects.append(
            dict(
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    new_objects = []
    handle_new_object = (
        lambda local_path, file_identifier, sha256, metadata: new_objects.append(
            dict(
                local_path=local_path,
                file_identifier=file_identifier,
                sha256=sha256,
                metadata=metadata,
            )
        )
    )

    modified_objects = []
    handle_modified_object = lambda local_path, file_identifier, new_sha256, old_sha256, metadata: modified_objects.append(
        dict(
            local_path=local_path,
            file_identifier=file_identifier,
            new_sha256=new_sha256,
            old_sha256=old_sha256,
            metadata=metadata,
        )
    )

    download_dir = "~/.objaverse-tests"
    base_download_dir = os.path.join(download_dir, "github")
    fs, path = fsspec.core.url_to_fs(base_download_dir)
    fs.makedirs(path, exist_ok=True)

    shutil.rmtree(os.path.join(path, "repos"), ignore_errors=True)
    out = github_downloader._process_repo(
        repo_id="mattdeitke/objaverse-xl-test-files-does-not-exist",
        fs=fs,
        base_dir=path,
        save_repo_format=None,
        expected_objects={
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx": "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example-2.fbx": "<fake-missing-object>",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb": "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj": "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
        },
        handle_found_object=handle_found_object,
        handle_modified_object=handle_modified_object,
        handle_missing_object=handle_missing_object,
        handle_new_object=handle_new_object,
        commit_hash="6928b08a2501aa7a4a4aabac1f888b66e7782056",
    )

    assert len(out) == 0
    assert len(found_objects) == 0
    assert len(missing_objects) == 4
    assert len(new_objects) == 0
    assert len(modified_objects) == 0


def test_github_download_cache():
    github_downloader = GitHubDownloader()
    objects = pd.DataFrame(
        [
            {
                "fileIdentifier": "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.fbx",
                "license": None,
                "sha256": "7037575f47816118e5a34e7c0da9927e1be7be3f5b4adfac337710822eb50fa9",
            },
            {
                "fileIdentifier": "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.glb",
                "license": None,
                "sha256": "04e6377317d6818e32c5cbd1951e76deb3641bbf4f6db6933046221d5fbf1c5c",
            },
            {
                "fileIdentifier": "https://github.com/mattdeitke/objaverse-xl-test-files/blob/6928b08a2501aa7a4a4aabac1f888b66e7782056/example.obj",
                "license": None,
                "sha256": "d2b9a5d7c47dc93526082c9b630157ab6bce4fd8669610d942176f4a36444e71",
            },
        ]
    )

    # remove the repos directory
    for save_repo_format in ["tar", "tar.gz", "zip", "files"]:
        repos_dir = "~/.objaverse-tests/github/repos"
        shutil.rmtree(os.path.expanduser(repos_dir), ignore_errors=True)

        out = github_downloader.download_objects(
            objects=objects,
            processes=1,
            download_dir="~/.objaverse-tests",
            save_repo_format=save_repo_format,  # type: ignore
        )
        assert len(out) == 3

        out = github_downloader.download_objects(
            objects=objects,
            processes=1,
            download_dir="~/.objaverse-tests",
            save_repo_format=save_repo_format,  # type: ignore
        )
        assert len(out) == 0


def test_annotations():
    downloaders = [
        GitHubDownloader(),
        SketchfabDownloader(),
        SmithsonianDownloader(),
        ThingiverseDownloader(),
    ]

    for downloader in downloaders:
        annotations_df = downloader.get_annotations()

        # make sure the columns are
        assert set(annotations_df.columns) == set(
            ["fileIdentifier", "source", "license", "fileType", "sha256", "metadata"]
        )


def test_download_objects():
    downloaders = [
        GitHubDownloader(),
        SketchfabDownloader(),
        SmithsonianDownloader(),
        ThingiverseDownloader(),
    ]

    download_dir = "~/.objaverse-tests"

    for downloader in downloaders:
        shutil.rmtree(os.path.expanduser(download_dir), ignore_errors=True)

        annotations_df = downloader.get_annotations()

        test_objects = annotations_df.head(n=2)

        out = downloader.download_objects(
            objects=test_objects,
            download_dir=download_dir,
            processes=2,
        )
        assert isinstance(out, dict), f"{out=}"
