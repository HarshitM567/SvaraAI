import requests
import json
def test_predict_local():
    # Simple format test. Ensure API is running locally before running this test.
    payload = {'text': 'Looking forward to the demo!'}
    resp = requests.post('http://127.0.0.1:8000/predict', json=payload, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert 'label' in data
