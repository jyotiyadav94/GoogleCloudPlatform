"""Integration test for the model.

Send a request to the model and check that the response is correct.
"""

import json

import requests

url = "http://app:8080/predictions/model"
headers = {"Content-Type": "application/json"}
data = {
    "instances": [
        ["uuid", "Hola me llamo Alex"],
        ["uuid", "The text is irrelevant, checking that the model works"],
    ]
}

print("Integration test: sending request to model...")

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print(f"Integration test: response {response.status_code} received.")
    predictions = response.json()
    if "predictions" not in predictions:
        print(f"Integration test: No predictions found. Response {response.text}.")
        raise Exception(f"Failed to get predictions: {response.text}")
    else:
        print(f"Integration test: Predictions found. Response {response.text}.")
else:
    raise Exception(f"Failed to get predictions: {response.text}")
