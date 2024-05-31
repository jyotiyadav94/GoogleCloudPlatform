"""
Builds the artifacts for the models and pushes them to the Artifact registry.

Usage: python scripts/build_push_container.py --models=opus-mt-de-en,opus-mt-en-de

Builds the artifacts for the models and pushes them to the Artifact registry.
1. It creates a .mar file
2. It creates a Dockerfile
3. It pushes the Docker image to the Artifact registry
4. It creates the model on Vertex AI (or creates a new version if it exists)
"""

import argparse
import datetime
import hashlib
import os
import subprocess

from dotenv import load_dotenv
from google.cloud import exceptions
from google.cloud.aiplatform import Model
from utils.cloud_storage import download_gcs_folder

load_dotenv("model.env")

# Get the models comma separated
parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, type=str)
parser.add_argument("--overwrite_mar", default=False, type=bool)
models = parser.parse_args().models.split(",")
overwrite_mar = parser.parse_args().overwrite_mar

# Download the models if they are not already from GCS
for model in models:
    # ------------------------------ Downloading ------------------------------
    model_local_path = os.path.join(os.environ["LOCAL_MODEL_DIR"], model, "raw")
    if not os.path.isdir(model_local_path):
        print(f"\nModel '{model}' Downloading...")
        download_gcs_folder(os.environ["MODELS_BUCKET"], model, model_local_path)
    else:
        print(f"\nModel '{model}' already downloaded. Skipping download.")

    # ---------------------------- Create .mar file ----------------------------
    # Add the extra files to build the .mar file
    # Remove the model since it's sent as serialized-file
    extra_files = ",".join(
        [
            os.path.join(model_local_path, file)
            for file in os.listdir(model_local_path)
            if file != "pytorch_model.bin"
        ]
    )

    mar_file = os.path.join(".", os.environ["LOCAL_MODEL_DIR"], model, f"{model}.mar")
    print(mar_file)
    if os.path.isfile(mar_file) and not overwrite_mar:
        print(f"\nMar file '{model}' already built. Skipping build.")
    else:
        print(f"\nBuilding {model}.mar file...")
        torchserve_command = [
            "torch-model-archiver",
            "--force",
            f"--model-name={model}",
            "--version=1.0",
            f"--serialized-file={model_local_path}/pytorch_model.bin",
            f"--extra-files={extra_files}",
            f"--export-path={os.path.join(os.environ['LOCAL_MODEL_DIR'], model)}",
            "--handler=handlers/handler.py",
        ]
        result = subprocess.run(torchserve_command, stdout=subprocess.PIPE, check=True)

    # ---------------------------- Build image ----------------------------
    # Creates random tag
    tag = hashlib.sha256(
        datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S").encode()
    ).hexdigest()
    model_image_uri = os.path.join(
        os.environ["ARTIFACT_REGISTRY_REPO_URI"], f"{model}:{tag}"
    ).lower()
    print(f"\nBuilding {model} Dockerfile...")
    docker_build_command = [
        "docker",
        "build",
        ".",
        f"--tag={model_image_uri}",
        "--build-arg",
        f"model_name={model}",
    ]
    result = subprocess.run(docker_build_command, stdout=subprocess.PIPE, check=True)

    # ---------------------------- Push image ----------------------------
    print(f"\nPushing {model} Dockerfile...")
    docker_push_command = [
        "docker",
        "push",
        model_image_uri,
    ]
    result = subprocess.run(docker_push_command, stdout=subprocess.PIPE, check=True)
    print(f"\nPushed {model_image_uri}")

    # ---------------------------- Create Vertex AI Model ----------------------------
    # Check if model exists
    new_model = False
    try:
        Model(
            project=os.environ["PROJECT_ID"],
            location=os.environ["REGION"],
            model_name=model,
        )
    except exceptions.NotFound:
        new_model = True

    # Uploads the model
    Model.upload(
        project=os.environ["PROJECT_ID"],
        location=os.environ["REGION"],
        serving_container_image_uri=os.path.join(model_image_uri),
        model_id=model,
        parent_model=model if not new_model else None,
        serving_container_predict_route="/predictions/model",
        serving_container_health_route="/ping",
        serving_container_ports=[8080],
        display_name=model,
        staging_bucket=os.environ["STAGING_BUCKET"],
    )

    print(f"Deployed model {model} to Vertex AI.")
