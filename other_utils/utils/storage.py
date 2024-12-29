from google.cloud.storage import Client, transfer_manager
from pathlib import Path
import os
import logging as logger
from utils.pipeline_utils import track_time


class GCS:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def __get_local_path(self, source_directory):
        # First, recursively get all files in `directory` as Path objects.
        directory_as_path_obj = Path(source_directory)
        paths = directory_as_path_obj.rglob("*")

        # Filter so the list only includes files, not directories themselves.
        file_paths = [path for path in paths if path.is_file()]

        # These paths are relative to the current working directory. Next, make them
        # relative to `directory`
        relative_paths = [path.relative_to(source_directory) for path in file_paths]

        # Finally, convert them all to strings.
        string_paths = [
            os.path.join(source_directory, str(path)) for path in relative_paths
        ]
        logger.debug(f"[String path] {string_paths}")
        return string_paths

    def download_bucket_with_transfer_manager(
        self,
        bucket_name,
        prefix,
        destination_directory="",
        workers=8,  # max_results=1000
    ):
        """Download all of the blobs in a bucket/dir, concurrently in a process pool."""

        # The maximum number of processes to use for the operation. The performance
        # impact of this value depends on the use case, but smaller files usually
        # benefit from a higher number of processes. Each additional process occupies
        # some CPU and memory resources until finished. Threads can be used instead
        # of processes by passing `worker_type=transfer_manager.THREAD`.
        # workers=8

        # The maximum number of results to fetch from bucket.list_blobs(). This
        # sample code fetches all of the blobs up to max_results and queues them all
        # for download at once. Though they will still be executed in batches up to
        # the processes limit, queueing them all at once can be taxing on system
        # memory if buckets are very large. Adjust max_results as needed for your
        # system environment, or set it to None if you are sure the bucket is not
        # too large to hold in memory easily.
        # max_results=1000

        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        # blob_names = [blob.name for blob in bucket.list_blobs(prefix=prefix, max_results=max_results) if blob.name[-1] != "/"]
        blob_names = [
            blob.name
            for blob in bucket.list_blobs(prefix=prefix)
            if blob.name[-1] != "/"
        ]

        results = transfer_manager.download_many_to_path(
            bucket,
            blob_names,
            destination_directory=destination_directory,
            max_workers=workers,
        )

        logger.info(
            f"{len(blob_names)} Downloaded Blobs from {os.path.join(bucket_name, prefix)}"
        )

        for name, result in zip(blob_names, results):
            # The results list is either `None` or an exception for each blob in
            # the input list, in order.

            if isinstance(result, Exception):
                logger.error(
                    "Failed to download {} due to exception: {}".format(name, result)
                )
            else:
                logger.debug(
                    "Downloaded {} to {}.".format(name, destination_directory + name)
                )

    def upload_many_blobs_with_transfer_manager(
        self, bucket_name, source_directory, workers=8
    ):
        """Upload every file in a list to a bucket, concurrently in a process pool."""

        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        filenames = self.__get_local_path(source_directory)

        results = transfer_manager.upload_many_from_filenames(
            bucket, filenames, max_workers=workers
        )

        for name, result in zip(filenames, results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to upload {} due to exception: {}".format(name, result)
                )
            else:
                logger.debug("Uploaded {} to {}.".format(name, bucket.name))

    @track_time
    def load_blobs(self, gcs_prefix):
        # if __name__ == "__main__":
        logger.info(f"Start loading blobs from {gcs_prefix}")
        self.download_bucket_with_transfer_manager(self.bucket_name, gcs_prefix)
        logger.info(f"Finished loading blobs from {gcs_prefix}")

    @track_time
    def save_blobs(self, local_prefix):
        # if __name__ == "__main__":
        logger.info(f"Start uploading blobs from {local_prefix}")
        self.upload_many_blobs_with_transfer_manager(self.bucket_name, local_prefix)
        logger.info(f"Finished uploading blobs from {local_prefix}")
