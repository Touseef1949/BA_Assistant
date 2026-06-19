"""Safe Locust scaffold for BA Assistant.

This file intentionally checks only Streamlit's root page and health endpoint.
It must not exercise any route or UI flow that can consume model credits.
"""

from locust import HttpUser, between, task


class BAAssistantSafeUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def root_page(self) -> None:
        self.client.get("/", name="root")

    @task(1)
    def streamlit_health(self) -> None:
        self.client.get("/_stcore/health", name="streamlit_health")
