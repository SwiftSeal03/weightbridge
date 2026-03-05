import json

class WeightData:
    """
    A unified representation of weight data for all backends.
    The format is:
    {
        "name": {
            "metadata": {
                "shard": [(l, r, w), ...],
                "dtype": torch.dtype,
            },
            "data": torch.Tensor,
        },
        ...
    }
    
    where [l, r) is the range of the local shard index on the dimension, w is the total width of the dimension.
    Data is a optionally flattened 1D tensor of shape \prod_{i=0}^{n-1} (r_i - l_i), where n is the number of dimensions.
    """
    def __init__(self, state_dict: dict[str, dict[str, ...]]):
        self.state_dict = state_dict
        self.sanity_check(self.state_dict)
        
    def __getitem__(self, key: str) -> dict[str, ...]:
        return self.state_dict[key]
    
    def __str__(self) -> str:
        metadata_dict = {k: v["metadata"] for k, v in self.state_dict.items()}
        return json.dumps(metadata_dict, indent=4)
    
    def sanity_check(self, state_dict: dict[str, dict[str, ...]]) -> None:
        for name, metadata in state_dict.items():
            numel = 1
            for l, r, w in metadata["shard"]:
                assert 0 <= l < r <= w, f"Invalid shard: {l, r, w} for {name}"
                assert w % (r - l) == 0 and l % (r - l) == 0, f"Invalid shard: {l, r, w} for {name}"
                numel *= r - l
            if t := metadata.get("data", None) is not None:
                assert t.numel() == numel, f"Invalid numel: {t.numel()} for {name}"
                assert t.dtype == metadata["dtype"], f"Invalid dtype: {t.dtype} for {name}"
                assert t.shape == (numel,), f"Invalid shape: {t.shape} for {name}"