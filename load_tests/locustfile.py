from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={"prompt": "Explain dynamic batching", "max_new_tokens": 64},
        )
