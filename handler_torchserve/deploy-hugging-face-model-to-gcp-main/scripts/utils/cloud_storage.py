"""Utility functions to upload/download files and folders from Google Cloud Storage."""

import os

from google.cloud import storage

client = None


def get_gcs_client() -> storage.Client:
    """Return a Google Cloud Storage client object."""
    global client
    if client is None:
        if "PROJECT_ID" not in os.environ:
            raise ValueError("The environment variable PROJECT_ID must be set.")
        client = storage.Client(project=os.environ["PROJECT_ID"])
    return client


def download_gcs_file(bucket_name: str, gcs_file_path: str, local_file_path: str):
    """Download a single file from Google Cloud Storage.

    Args:
        bucket_name (str): Bucket name
        gcs_file_path (str): Path to the file in the bucket (e.g. "folder/file.txt")
        local_file_path (str): Local path to save the file (e.g. "data/file.txt")
    """
    # Initialize a client object
    storage_client = get_gcs_client()

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # Get the blob object
    blob = bucket.blob(gcs_file_path)

    # Download the blob to a file
    blob.download_to_filename(local_file_path)

    print(f"Blob {gcs_file_path} downloaded to {local_file_path}.")


def download_gcs_folder(bucket_name: str, gcs_folder: str, local_folder: str):
    """Download a folder from Google Cloud Storage. Excluding subdirectories.

    Args:
        bucket_name (str): Bucket name
        gcs_folder (str): Path the folder on the bucket (e.g. "folder/")
        local_folder (str): Path to the local folder (e.g. "data/")
    """
    # Creates the directory if it doesn't exist
    os.makedirs(local_folder, exist_ok=True)

    # Initialize a client object
    storage_client = get_gcs_client()

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # Get all the files in the subdirectory
    blobs = bucket.list_blobs(prefix=gcs_folder)

    for blob in blobs:
        # Download only files (not subdirectories)
        if "." in blob.name:
            file_path = os.path.join(local_folder, os.path.basename(blob.name))
            blob.download_to_filename(file_path)


def upload_folder_to_gcs(
    bucket_name: str, gcs_folder_path: str, local_folder_path: str
):
    """
    Upload a full folder to Google Cloud Storage.

    Args:
        bucket_name (str): Name of the bucket to upload to.
        gcs_folder_path (str): Path to the folder on Google Cloud Storage
            (does not include the bucket name).
        local_folder_path (str): Path to the local folder to upload.


    """
    # Create a client object for Google Cloud Storage
    client = get_gcs_client()

    # Get the bucket object
    bucket = client.bucket(bucket_name)

    # Create the GCS folder if it does not exist
    gcs_folder = bucket.blob(gcs_folder_path)
    gcs_folder.upload_from_string("")

    # Upload each file in the local folder to the GCS folder
    uploaded_files = []
    for file_path in os.listdir(local_folder_path):
        local_file_path = os.path.join(local_folder_path, file_path)
        if os.path.isfile(local_file_path):
            blob_name = os.path.join(gcs_folder_path, file_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            uploaded_files.append(blob_name)
