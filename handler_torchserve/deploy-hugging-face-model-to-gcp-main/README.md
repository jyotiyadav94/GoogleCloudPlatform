# deploy-hugging-face-model-to-gcp
Repository with the tools to upload a Hugging Face model to the Model Registry of Vertex AI to make batch/online predictions

## Local development instructions

1. Run `pip install requirements.txt`
2. Set env variables in model.env
3. Run `scripts/build_push_container.py` to build the image, push it to the registry and create the models in Vertex AI
## Tests
### Integration test
1. Run `export IMAGE=your-built-image` (Check `scripts/build_push_container.py` for the image name)
2. Run `docker-compose -f tests/integration/docker-compose.yaml up --exit-code-from test --renew-anon-volumes && docker compose -f tests/integration/docker-compose.yaml down -v`
