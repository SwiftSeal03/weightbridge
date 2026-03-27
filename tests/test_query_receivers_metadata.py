"""Test that WeightReceiverController._query_receivers_metadata works with 2 WeightReceivers."""

import time

from fastapi import FastAPI

from wbridge.frontend import WeightReceiver, WeightReceiverController
from wbridge.utils.data import WeightData


def test_query_receivers_metadata():
    """`_query_receivers_metadata` returns one dict per worker (rank, metadata, state, …), sorted by rank."""
    app = FastAPI()
    controller = WeightReceiverController(app, worker_num=0)
    controller.set_worker_num(2)

    empty = WeightData(meta_dict={})
    receiver0 = WeightReceiver(controller_ipc_name=controller.ipc_name, rank=0, metadata=empty)
    receiver1 = WeightReceiver(controller_ipc_name=controller.ipc_name, rank=1, metadata=empty)

    # Give receivers time to connect and start polling
    time.sleep(0.5)

    results = controller._query_receivers_metadata()
    assert len(results) == 2, f"Expected 2 responses, got {results}"
    assert [r["rank"] for r in results] == [0, 1]
    assert all(r["metadata"] == {} for r in results)


if __name__ == "__main__":
    test_query_receivers_metadata()
    print("test_query_receivers_metadata passed")
