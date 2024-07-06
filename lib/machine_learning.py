import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class MachineLearning:
    """Machine learning model wrapper for VGG19 with batch normalization.

    Attributes:
        model (torch.nn.Module): The VGG19 model with custom classifier.
        transform (torchvision.transforms.Compose): Image transformation pipeline.
    """

    def __init__(self, model_type: str) -> None:
        """Initializes the MachineLearning model.

        Args:
            model_type (str): Type of the model to load.
            model_path (str): Path to the pretrained model file.

        Raises:
            ValueError: If an unsupported model type is provided.
        """
        if model_type != "vgg19_bn":
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model = models.vgg19_bn(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=16)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])

    def load_model(self, model_path: str) -> None:
        """Loads the model state dictionary from the given path.

        Args:
            model_path (str): Path to the model state dictionary file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file at '{model_path}' does not exist.")

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        print(f"Loading model from {model_path}")

    def inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Performs inference on an image tensor.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Predicted class logits.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        return output
