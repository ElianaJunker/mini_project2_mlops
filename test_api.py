import requests

BASE_URL = "http://127.0.0.1:8000"

def test_predict():
    url = f"{BASE_URL}/predict"
    test_data = {"data": [[10.1, 2.5, 7.4, 0.2]]}
    response = requests.post(url, json=test_data)
    assert response.status_code == 200, f"Failed predict test: {response.text}"
    print("Prediction test successful:", response.json())

def test_update_model(new_version="2"):
    url = f"{BASE_URL}/update-model?new_version={new_version}"  # Pass new_version as query param
    response = requests.post(url)
    assert response.status_code == 200, f"Failed update model test: {response.text}"
    print("Update model test successful:", response.json())

if __name__ == "__main__":
    test_predict()
    test_update_model()
    test_predict()
