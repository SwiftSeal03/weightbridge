"""Frontend API for weight transfer."""

from wbridge.frontend.receiver import WeightReceiver, WeightReceiverController
from wbridge.frontend.sender import WeightSender

__all__ = ["WeightReceiver", "WeightSender", "WeightReceiverController"]
