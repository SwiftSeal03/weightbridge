"""HTTP route for receiver world size (WeightReceiverController)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wbridge.frontend import WeightReceiverController


def test_receiver_world_route():
    app = FastAPI()
    controller = WeightReceiverController(app)
    controller.set_worker_num(2)
    client = TestClient(app)
    resp = client.get("/wbridge/receiver_world")
    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "world_size": 2}


if __name__ == "__main__":
    test_receiver_world_route()
    print("test_receiver_world_route passed")
