from fastapi.testclient import TestClient
from backend.app import app

def test_backend():
    print("Initializing TestClient (triggers lifespan events)...")
    with TestClient(app) as client:
        print("\n--- Testing GET /health ---")
        health = client.get("/health")
        print("Status Code:", health.status_code)
        print("Response:", health.json())
        assert health.status_code == 200

        print("\n--- Testing GET /features/schema ---")
        schema = client.get("/features/schema")
        print("Status Code:", schema.status_code)
        print("Response Keys:", schema.json().keys())
        assert schema.status_code == 200

        print("\n--- Testing POST /predict (Valid URL) ---")
        predict1 = client.post("/predict", json={"url": "https://example.com/login"})
        print("Status Code:", predict1.status_code)
        if predict1.status_code == 200:
            res = predict1.json()
            print("Result Label:", res["result"]["label"])
            print("Probability:", res["result"]["probability"])
            print("Is Punycode:", res["url_analysis"]["is_punycode"])
            print("Top Features:", [f["feature"] for f in res["top_features"]])
        else:
            import json
            with open("error_output.json", "w") as f:
                json.dump(predict1.json(), f)
            print("Error saved to error_output.json")
        assert predict1.status_code == 200

        print("\n--- Testing POST /predict (IDN Homograph) ---")
        predict2 = client.post("/predict", json={"url": "https://xn--pple-43d.com/login"})
        print("Status Code:", predict2.status_code)
        if predict2.status_code == 200:
            res = predict2.json()
            print("Result Label:", res["result"]["label"])
            print("Decoded URL:", res["url_analysis"]["url_decoded"])
            print("Is Punycode:", res["url_analysis"]["is_punycode"])
            print("Warning:", res["url_analysis"]["punycode_warning"])
        assert predict2.status_code == 200

if __name__ == "__main__":
    test_backend()
    print("\nAll tests passed successfully!")
