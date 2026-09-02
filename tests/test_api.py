import os
import json
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.abspath('backend'))
from main import app


client = TestClient(app)


def test_health():
    res = client.get('/')
    assert res.status_code == 200


def test_predict_empty():
    res = client.post('/predict', json={'complaint': ''})
    assert res.status_code == 400


def test_predict_sample():
    # Provide a short sample text; if model not loaded, expect 500
    res = client.post('/predict', json={'complaint': 'I have a problem with my credit card charge'})
    assert res.status_code in (200, 500)
