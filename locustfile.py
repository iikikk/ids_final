from locust import HttpUser, task, between

class MicroserviceUser(HttpUser):
    # Simulates a wait time between consecutive tasks
    wait_time = between(0.001, 0.005)

    @task
    def predict(self):
        # Payload for the prediction request
        payload = {"text": "theres something wrong when a girl wins wayne rooney street striker"}
        # Sends a POST request to the /predict endpoint
        self.client.post("/predict", json=payload)
