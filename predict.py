import os
from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download


# ==================== CONFIG ====================

# Use GPU when available; otherwise use CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# IMPORTANT:
# Replace "YOUR_HF_USERNAME" with your actual Hugging Face username.
#
# Example:
# HF_REPO_ID = "john123/Skin-Disease-Diagnosis-Model"
#
HF_REPO_ID = "somtochukwu901/Skin-Disease-Application"

HF_FILENAME = "model.pt"


# IMPORTANT:
# The order MUST be identical to the class-index order used during training.
#
# If torchvision ImageFolder was used during training, classes are normally
# assigned in alphabetical order.
CLASS_NAMES = [
    "Eczema",
    "Fungal",
    "Others",
    "Scabies",
    "Dermatitis",
]


# A ResNet50 checkpoint should normally be substantially larger than this.
# This also helps detect a Git LFS pointer or incomplete download.
MIN_EXPECTED_BYTES = 5 * 1024 * 1024


# ==================== MODEL ARCHITECTURE ====================

class ResNet50Model(nn.Module):
    """
    ResNet50 architecture used by the trained model.

    The checkpoint is expected to contain keys beginning with:
        backbone.*

    The classifier architecture is:

        Dropout
        Linear(2048 -> 512)
        BatchNorm1d(512)
        ReLU
        Dropout
        Linear(512 -> num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ==================== DOWNLOAD ====================

def get_model_path() -> str:
    """
    Download model.pt from Hugging Face Hub and return its local path.

    The Hugging Face cache is reused automatically.

    For a private repository, set:
        HF_TOKEN

    as an environment variable.
    """

    if not HF_REPO_ID or HF_REPO_ID.startswith(
        "YOUR_HF_USERNAME/"
    ):
        raise ValueError(
            "HF_REPO_ID has not been configured. "
            "Replace 'YOUR_HF_USERNAME' with your actual "
            "Hugging Face username in predict.py."
        )

    token = os.environ.get("HF_TOKEN")

    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{HF_FILENAME}' from "
            f"Hugging Face repository '{HF_REPO_ID}'. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    # Check that the returned file actually exists.
    if not os.path.isfile(path):

        print(
            f"Cached model path is missing or broken. "
            f"Attempting a fresh download: {path}",
            flush=True,
        )

        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                token=token,
                force_download=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Fresh model download failed. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"hf_hub_download returned a path that does not exist: "
            f"{path}"
        )

    return path


# ==================== WEIGHTS VALIDATION ====================

def _check_weights_file(path: str) -> None:
    """
    Verify that the downloaded model looks like a real checkpoint.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file does not exist: {path}"
        )

    size = os.path.getsize(path)

    with open(path, "rb") as f:
        head = f.read(128)

    # Detect Git LFS pointer files.
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"'{HF_FILENAME}' is a Git LFS pointer file "
            f"({size} bytes), not the actual model weights.\n\n"
            "The model needs to be uploaded correctly to Hugging Face. "
            "Make sure the actual model file is available in the repository."
        )

    # Detect suspiciously small files.
    if size < MIN_EXPECTED_BYTES:
        raise RuntimeError(
            f"'{HF_FILENAME}' is only "
            f"{size / 1024:.2f} KB.\n\n"
            "That is too small for a ResNet50 checkpoint and usually "
            "means the upload/download is incomplete."
        )

    print(
        f"Model file OK: {path} "
        f"({size / (1024 * 1024):.2f} MB)",
        flush=True,
    )


# ==================== CHECKPOINT READING ====================

def _read_checkpoint(path: str):
    """
    Read a PyTorch checkpoint.

    First attempts the safer weights_only loader.

    If that fails, falls back to normal pickle loading. The fallback should
    only be used with checkpoints that you trust and created yourself.
    """

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        return checkpoint

    except Exception as safe_error:

        print(
            "weights_only=True failed: "
            f"{type(safe_error).__name__}: {safe_error}",
            flush=True,
        )

        print(
            "Trying weights_only=False. "
            "Only do this with a trusted checkpoint.",
            flush=True,
        )

        try:
            return torch.load(
                path,
                map_location=device,
                weights_only=False,
            )

        except Exception as full_error:

            raise RuntimeError(
                "The model checkpoint could not be loaded.\n\n"
                f"Safe loader error: "
                f"{type(safe_error).__name__}: {safe_error}\n\n"
                f"Full loader error: "
                f"{type(full_error).__name__}: {full_error}"
            ) from full_error


# ==================== STATE DICT EXTRACTION ====================

def _extract_state_dict(checkpoint) -> dict:
    """
    Extract a state_dict from common checkpoint formats.

    Supported examples:

        state_dict

        {
            "state_dict": {...}
        }

        {
            "model_state_dict": {...}
        }

        {
            "model": {...}
        }
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dictionary checkpoint, "
            f"got {type(checkpoint).__name__}."
        )

    # Common wrappers.
    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        inner = checkpoint.get(key)

        if isinstance(inner, dict):
            checkpoint = inner
            break

    if not checkpoint:
        raise ValueError(
            "The checkpoint contains an empty state dictionary."
        )

    cleaned = {}

    for key, value in checkpoint.items():

        # DataParallel checkpoints often contain "module."
        # at the beginning of every parameter name.
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ==================== MODEL LOADING ====================

def load_model(model_path: str) -> nn.Module:
    """
    Load the trained model and return it in evaluation mode.

    Raises a clear exception when the checkpoint does not match
    the expected architecture.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Current working directory: {os.getcwd()}"
        )

    _check_weights_file(model_path)

    checkpoint = _read_checkpoint(model_path)


    # --------------------------------------------------
    # CASE 1: Whole model object
    # --------------------------------------------------

    if isinstance(checkpoint, nn.Module):

        checkpoint.eval()
        checkpoint.to(device)

        print(
            "Loaded a complete PyTorch model object.",
            flush=True,
        )

        return checkpoint


    # --------------------------------------------------
    # CASE 2: State dictionary
    # --------------------------------------------------

    if isinstance(checkpoint, dict):

        state = _extract_state_dict(checkpoint)

        # Final layer of our custom classifier.
        fc_key = "backbone.fc.5.weight"

        if fc_key not in state:

            sample_keys = list(state.keys())[:10]

            raise KeyError(
                f"Expected checkpoint key '{fc_key}' was not found.\n\n"
                f"First checkpoint keys:\n{sample_keys}\n\n"
                "This usually means that the checkpoint architecture "
                "does not match ResNet50Model."
            )

        final_layer = state[fc_key]

        if not hasattr(final_layer, "shape"):
            raise TypeError(
                f"Checkpoint value '{fc_key}' does not have a shape."
            )

        ckpt_classes = final_layer.shape[0]

        if ckpt_classes != len(CLASS_NAMES):

            raise ValueError(
                f"Checkpoint has {ckpt_classes} output classes, "
                f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes.\n\n"
                f"CLASS_NAMES = {CLASS_NAMES}\n\n"
                "The class names and their order must exactly match "
                "the labels used during model training."
            )

        model = ResNet50Model(
            num_classes=ckpt_classes
        )

        try:

            model.load_state_dict(
                state,
                strict=True,
            )

        except RuntimeError as e:

            raise RuntimeError(
                "The checkpoint was found, but its weights do not "
                "match the expected ResNet50Model architecture.\n\n"
                f"PyTorch error:\n{e}"
            ) from e

        model.eval()
        model.to(device)

        print(
            f"Loaded ResNet50 weights successfully "
            f"({ckpt_classes} classes).",
            flush=True,
        )

        return model


    # --------------------------------------------------
    # Unknown checkpoint
    # --------------------------------------------------

    raise ValueError(
        "Unrecognized checkpoint format: "
        f"{type(checkpoint).__name__}"
    )


# ==================== PREPROCESSING ====================

# These transforms must match the validation/test transforms used
# during training.

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ==================== PREDICTION ====================

def predict_image(
    image: Union[str, Image.Image],
    model: nn.Module,
):
    """
    Predict the skin-disease class.

    Parameters
    ----------
    image:
        A file path or PIL.Image.Image.

    model:
        Loaded PyTorch model.

    Returns
    -------
    tuple:
        (predicted_class, confidence_percentage)
    """

    # Handle file path.
    if not isinstance(image, Image.Image):

        try:
            image = Image.open(image)

        except Exception as e:

            raise ValueError(
                f"Could not open image: {e}"
            ) from e

    # Correct EXIF rotation and force RGB.
    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")


    # Apply preprocessing.
    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)


    # Inference.
    with torch.inference_mode():

        outputs = model(input_tensor)

        # Some models may return a tuple/list.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if not torch.is_tensor(outputs):
            raise TypeError(
                "Model output is not a PyTorch tensor."
            )

        if outputs.ndim != 2 or outputs.shape[0] != 1:
            raise ValueError(
                f"Unexpected model output shape: "
                f"{tuple(outputs.shape)}. "
                "Expected [1, number_of_classes]."
            )

        probabilities = torch.softmax(
            outputs[0],
            dim=0,
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=0,
        )


    # Convert prediction index to class name.
    index = predicted_idx.item()

    if index < 0 or index >= len(CLASS_NAMES):

        raise IndexError(
            f"Predicted class index {index} is outside "
            f"CLASS_NAMES range 0-{len(CLASS_NAMES) - 1}."
        )

    prediction = CLASS_NAMES[index]

    confidence_percent = (
        confidence.item() * 100.0
    )


    # Optional debugging output.
    if os.environ.get("SKIN_APP_DEBUG"):

        print(
            "\nPrediction probabilities:",
            flush=True,
        )

        for name, probability in zip(
            CLASS_NAMES,
            probabilities.tolist(),
        ):
            print(
                f"   {name}: "
                f"{probability * 100:.2f}%",
                flush=True,
            )


    return prediction, confidence_percent


# ==================== LOCAL TEST ====================

if __name__ == "__main__":

    print(
        f"Using device: {device}",
        flush=True,
    )

    try:

        model_path = get_model_path()

        print(
            f"Model path: {model_path}",
            flush=True,
        )

        net = load_model(
            model_path
        )

        test_image_path = "test_image.jpg"

        if not os.path.isfile(test_image_path):

            raise FileNotFoundError(
                f"Test image not found: "
                f"{test_image_path}"
            )

        label, confidence = predict_image(
            test_image_path,
            net,
        )

        print(
            f"Diagnosis: {label}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:

        print(
            "\n❌ Prediction test failed:",
            flush=True,
        )

        print(
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise
