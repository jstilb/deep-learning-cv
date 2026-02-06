"""Model architectures: custom CNN, transfer learning, and Grad-CAM."""

from src.models.cnn import CustomCNN
from src.models.transfer import TransferLearningModel

__all__ = ["CustomCNN", "TransferLearningModel"]
