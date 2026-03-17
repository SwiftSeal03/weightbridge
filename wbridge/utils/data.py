import json

import torch


class WeightData:
    """
    A unified representation of weight metadata for receivers.
    The format is:
    {
        "name": {
            "metadata": {
                "shard": [[(l, r, w), ...], [(l, r, w), ...], ...],
                "dtype": torch.dtype,
            },
            "data": torch.Tensor | None,  # optional, receivers only need metadata
        },
        ...
    }

    where [l, r) is the range of the local shard index on the dimension, w is the total width.
    Receivers only need metadata (shard + dtype); data is optional for senders.
    """

    def __init__(self, state_dict: dict[str, dict[str, ...]]):
        self.state_dict = state_dict
        self.sanity_check(self.state_dict)

    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.state_dict[key]

    def to_metadata_dict(self) -> dict[str, dict]:
        """JSON-serializable metadata only (shard + dtype string)."""
        return {
            k: {"shard": v["metadata"]["shard"], "dtype": str(v["metadata"]["dtype"])}
            for k, v in self.state_dict.items()
        }

    def __str__(self) -> str:
        return json.dumps(self.to_metadata_dict(), indent=4)

    def sanity_check(self, state_dict: dict[str, dict[str, ...]]) -> None:
        for name, value in state_dict.items():
            meta = value["metadata"]
            dtype = meta["dtype"]
            numel = 1
            for l, r, w in meta["shard"]:
                assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
                numel *= r - l
            if (t := value.get("data")) is not None:
                nbytes = numel * dtype.itemsize
                assert t.dtype == torch.uint8, f"Invalid dtype: {t.dtype} for {name}"
                assert t.nbytes == nbytes, f"Invalid nbytes: {t.nbytes} for {name}, expected {nbytes}"
                assert len(t.shape) == 1, f"Invalid shape: {t.shape} for {name}"
