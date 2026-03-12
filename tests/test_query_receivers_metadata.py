"""Test that WeightReceiverController._query_receivers_metadata works with 2 WeightReceivers."""

import time

import pytest
from fastapi import FastAPI

from wbridge.frontend import WeightReceiver, WeightReceiverController


def test_query_receivers_metadata():
    """Create WeightReceiverController and 2 WeightReceivers, verify _query_receivers_metadata returns ranks."""
    app = FastAPI()
    controller = WeightReceiverController(app, worker_num=0)
    controller.set_worker_num(2)

    receiver0 = WeightReceiver(controller_ipc_name=controller.ipc_name, rank=0)
    receiver1 = WeightReceiver(controller_ipc_name=controller.ipc_name, rank=1)

    # Give receivers time to connect and start polling
    time.sleep(0.5)

    ranks = controller._query_receivers_metadata()
    assert len(ranks) == 2, f"Expected 2 ranks, got {ranks}"
    assert set(ranks) == {0, 1}, f"Expected ranks {{0, 1}}, got {ranks}"


if __name__ == "__main__":
    test_query_receivers_metadata()
    print("test_query_receivers_metadata passed")
