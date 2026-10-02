"""
Comprehensive Model Testing Script
Tests whether the skin-disease model loads correctly,
produces valid predictions, and is ready for application deployment.

IMPORTANT:
Random images only test technical inference.
They do NOT measure medical accuracy.
For accuracy testing, use labeled images that were not used during training.
"""

import os
import sys
import time
import traceback

import numpy as np
import torch
import torchvision
from PIL import Image

from predict import (
    CLASS_NAMES,
    device,
    get_model_path,
    load_model,
    predict_image,
    transform,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEMP_IMAGE_PATH = "test_image_temp.jpg"

passed_tests = 0
failed_tests = 0
warning_tests = 0


def test_pass(message: str):
    global passed_tests
    passed_tests += 1
    print(f"  ✅ {message}")


def test_fail(message: str):
    global failed_tests
    failed_tests += 1
    print(f"  ❌ {message}")


def test_warning(message: str):
    global warning_tests
    warning_tests += 1
    print(f"  ⚠️ {message}")


def section(number: int, title: str):
    print()
    print("=" * 70)
    print(f"TEST {number}: {title}")
    print("=" * 70)


print("=" * 70)
print("🔬 DERMA SCAN - MODEL TESTING & VERIFICATION")
print("=" * 70)


# ============================================================
# TEST 1: DEPENDENCIES
# ============================================================

section(1, "Checking Dependencies")

try:
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ TorchVision: {torchvision.__version__}")
    print(f"  ✓ Pillow: {Image.__version__}")
    print(f"  ✓ NumPy: {np.__version__}")
    print(f"  ✓ Device: {device}")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ℹ️ CUDA not available; inference will use CPU.")

    test_pass("All required dependencies loaded successfully.")

except Exception as e:
    test_fail(f"Dependency error: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# TEST 2: MODEL CONFIGURATION
# ============================================================

section(2, "Checking Model Configuration")

try:
    print(f"  Hugging Face repository: configured through predict.py")
    print(f"  Model filename: model.pt")
    print(f"  Number of classes: {len(CLASS_NAMES)}")
    print(f"  Classes: {', '.join(CLASS_NAMES)}")

    if len(CLASS_NAMES) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    expected_classes = {
        "Eczema",
        "Fungal",
        "Others",
        "Scabies",
        "Dermatitis",
    }

    if set(CLASS_NAMES) != expected_classes:
        test_warning(
            "CLASS_NAMES differs from the expected five-class set. "
            "Verify this is intentional."
        )
    else:
        print("  ✓ Expected five skin-disease classes detected.")

    test_pass("Model configuration is readable.")

except Exception as e:
    test_fail(f"Configuration error: {type(e).__name__}: {e}")


# ============================================================
# TEST 3: DOWNLOAD / LOCATE MODEL
# ============================================================

section(3, "Locating Model Checkpoint")

model_path = None

try:
    model_path = get_model_path()

    print(f"  Model path: {model_path}")
    print(f"  File exists: {os.path.isfile(model_path)}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    file_size_mb = (
        os.path.getsize(model_path)
        / (1024 * 1024)
    )

    print(f"  File size: {file_size_mb:.2f} MB")

    if file_size_mb < 5:
        raise RuntimeError(
            "Model file is suspiciously small."
        )

    test_pass("Model checkpoint is available.")

except Exception as e:
    test_fail(
        f"Could not locate/download model: "
        f"{type(e).__name__}: {e}"
    )

    print("\nThe most likely causes are:")
    print("  1. HF_REPO_ID is incorrect in predict.py")
    print("  2. model.pt is missing from the Hugging Face repository")
    print("  3. The Hugging Face repository is private and HF_TOKEN is missing")
    print("  4. Internet access is unavailable")

    sys.exit(1)


# ============================================================
# TEST 4: LOAD MODEL
# ============================================================

section(4, "Loading Model")

model = None

try:
    model = load_model(model_path)

    if model is None:
        raise RuntimeError(
            "load_model() returned None."
        )

    print(
        f"  Model type: {type(model).__name__}"
    )

    model_device = next(
        model.parameters()
    ).device

    print(
        f"  Model device: {model_device}"
    )

    print(
        f"  Model mode: "
        f"{'eval' if not model.training else 'train'}"
    )

    if model.training:
        test_warning(
            "Model is in training mode. "
            "load_model() should normally return eval mode."
        )
    else:
        print("  ✓ Model is in evaluation mode.")

    test_pass("Model loaded successfully.")

except Exception as e:
    test_fail(
        f"Model loading failed: "
        f"{type(e).__name__}: {e}"
    )

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# TEST 5: MODEL ARCHITECTURE
# ============================================================

section(5, "Checking Model Architecture")

try:
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"  Trainable parameters: "
        f"{trainable_params:,}"
    )

    leaf_layers = len(
        [
            module
            for module in model.modules()
            if not list(module.children())
        ]
    )

    print(
        f"  Leaf layers/modules: "
        f"{leaf_layers}"
    )

    model_name = type(model).__name__

    print(
        f"  Model class: {model_name}"
    )

    # Check whether the expected ResNet50 backbone exists.
    if hasattr(model, "backbone"):

        backbone_name = type(
            model.backbone
        ).__name__

        print(
            f"  Backbone: {backbone_name}"
        )

        if backbone_name == "ResNet":
            test_pass(
                "ResNet backbone detected."
            )
        else:
            test_warning(
                f"Expected ResNet backbone, "
                f"found {backbone_name}."
            )

    else:

        test_warning(
            "Model does not expose a 'backbone' attribute. "
            "This may be valid for a full-model checkpoint."
        )

    test_pass("Model architecture inspection completed.")

except Exception as e:

    test_warning(
        f"Architecture inspection failed: "
        f"{type(e).__name__}: {e}"
    )


# ============================================================
# TEST 6: CREATE RANDOM TEST IMAGE
# ============================================================

section(6, "Creating Technical Test Image")

try:

    test_array = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8,
    )

    test_pil_image = Image.fromarray(
        test_array
    )

    test_pil_image.save(
        TEMP_IMAGE_PATH,
        format="JPEG",
        quality=95,
    )

    print(
        f"  Test image: {TEMP_IMAGE_PATH}"
    )

    print(
        f"  Size: {test_pil_image.size}"
    )

    print(
        f"  Mode: {test_pil_image.mode}"
    )

    if test_pil_image.size != (224, 224):
        raise ValueError(
            "Generated test image has incorrect dimensions."
        )

    test_pass(
        "Random technical test image created."
    )

    print(
        "  ℹ️ This image is only for testing the software pipeline."
    )

except Exception as e:

    test_fail(
        f"Could not create test image: "
        f"{type(e).__name__}: {e}"
    )

    sys.exit(1)


# ============================================================
# TEST 7: PREPROCESSING
# ============================================================

section(7, "Checking Image Preprocessing")

try:

    image = Image.open(
        TEMP_IMAGE_PATH
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    print(
        f"  Input tensor shape: "
        f"{tuple(input_tensor.shape)}"
    )

    print(
        f"  Input tensor dtype: "
        f"{input_tensor.dtype}"
    )

    print(
        f"  Input tensor device: "
        f"{input_tensor.device}"
    )

    expected_shape = (
        1,
        3,
        224,
        224,
    )

    if tuple(input_tensor.shape) != expected_shape:
        raise ValueError(
            f"Expected tensor shape "
            f"{expected_shape}, "
            f"got {tuple(input_tensor.shape)}"
        )

    if not torch.isfinite(
        input_tensor
    ).all():
        raise ValueError(
            "Input tensor contains NaN or infinity."
        )

    test_pass(
        "Image preprocessing is working correctly."
    )

except Exception as e:

    test_fail(
        f"Preprocessing test failed: "
        f"{type(e).__name__}: {e}"
    )

    traceback.print_exc()
    sys.exit(1)


# ============================================================
# TEST 8: SINGLE INFERENCE
# ============================================================

section(8, "Testing Model Inference")

try:

    prediction, confidence = predict_image(
        TEMP_IMAGE_PATH,
        model,
    )

    print(
        f"  Input: {TEMP_IMAGE_PATH}"
    )

    print(
        f"  Prediction: {prediction}"
    )

    print(
        f"  Confidence: {confidence:.2f}%"
    )

    if not isinstance(
        prediction,
        str,
    ):
        raise TypeError(
            "Prediction must be a string."
        )

    if not isinstance(
        confidence,
        (float, int),
    ):
        raise TypeError(
            "Confidence must be numeric."
        )

    if not (
        0 <= float(confidence) <= 100
    ):
        raise ValueError(
            "Confidence must be between 0 and 100."
        )

    if prediction not in CLASS_NAMES:
        raise ValueError(
            f"Prediction '{prediction}' "
            f"is not in CLASS_NAMES."
        )

    test_pass(
        "Single-image inference succeeded."
    )

except Exception as e:

    test_fail(
        f"Inference failed: "
        f"{type(e).__name__}: {e}"
    )

    traceback.print_exc()
    sys.exit(1)


# ============================================================
# TEST 9: MULTIPLE INFERENCE RUNS
# ============================================================

section(9, "Testing Multiple Predictions")

try:

    predictions = []
    confidences = []

    for i in range(3):

        random_array = np.random.randint(
            0,
            256,
            (224, 224, 3),
            dtype=np.uint8,
        )

        random_image = Image.fromarray(
            random_array
        )

        random_image.save(
            TEMP_IMAGE_PATH,
            format="JPEG",
            quality=95,
        )

        prediction_i, confidence_i = (
            predict_image(
                TEMP_IMAGE_PATH,
                model,
            )
        )

        predictions.append(
            prediction_i
        )

        confidences.append(
            float(confidence_i)
        )

        print(
            f"  Run {i + 1}: "
            f"{prediction_i:12} "
            f"(Confidence: {confidence_i:6.2f}%)"
        )

    avg_confidence = float(
        np.mean(confidences)
    )

    min_confidence = min(
        confidences
    )

    max_confidence = max(
        confidences
    )

    print()
    print(
        f"  Average confidence: "
        f"{avg_confidence:.2f}%"
    )

    print(
        f"  Confidence range: "
        f"{min_confidence:.2f}% - "
        f"{max_confidence:.2f}%"
    )

    test_pass(
        "Multiple inference runs completed."
    )

except Exception as e:

    test_fail(
        f"Multiple inference test failed: "
        f"{type(e).__name__}: {e}"
    )

    traceback.print_exc()


# ============================================================
# TEST 10: RAW OUTPUT / PROBABILITIES
# ============================================================

section(10, "Checking Output Distribution")

try:

    image = Image.open(
        TEMP_IMAGE_PATH
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(
            input_tensor
        )

        if isinstance(
            outputs,
            (tuple, list),
        ):
            outputs = outputs[0]

        probabilities = torch.softmax(
            outputs[0],
            dim=0,
        )

    print(
        f"  Model output shape: "
        f"{tuple(outputs.shape)}"
    )

    print(
        f"  Expected output classes: "
        f"{len(CLASS_NAMES)}"
    )

    if outputs.ndim != 2:
        raise ValueError(
            "Model output must have two dimensions."
        )

    if outputs.shape[0] != 1:
        raise ValueError(
            "Expected batch size of 1."
        )

    if outputs.shape[1] != len(
        CLASS_NAMES
    ):
        raise ValueError(
            f"Model produces {outputs.shape[1]} "
            f"outputs, but CLASS_NAMES contains "
            f"{len(CLASS_NAMES)} classes."
        )

    print(
        f"  Output range: "
        f"[{outputs.min().item():.4f}, "
        f"{outputs.max().item():.4f}]"
    )

    print()
    print("  Class probabilities:")

    for cls_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):

        prob_pct = (
            probability.item() * 100
        )

        bar_length = min(
            50,
            int(prob_pct / 2),
        )

        bar = (
            "█" * bar_length
            + "░" * (50 - bar_length)
        )

        print(
            f"    {cls_name:12} "
            f"{prob_pct:6.2f}% |{bar}|"
        )

    total_probability = (
        probabilities.sum().item()
    )

    print()
    print(
        f"  Total probability: "
        f"{total_probability:.6f}"
    )

    if not np.isclose(
        total_probability,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Softmax probabilities do not sum to 1."
        )

    if not torch.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probabilities contain NaN or infinity."
        )

    test_pass(
        "Output distribution is valid."
    )

except Exception as e:

    test_fail(
        f"Output distribution test failed: "
        f"{type(e).__name__}: {e}"
    )

    traceback.print_exc()


# ============================================================
# TEST 11: REAL SAMPLE IMAGES
# ============================================================

section(11, "Testing Real Sample Images")

try:

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
    )

    sample_images = [
        filename
        for filename in os.listdir(".")
        if filename.lower().endswith(
            image_extensions
        )
        and filename != TEMP_IMAGE_PATH
    ]

    if not sample_images:

        test_warning(
            "No real sample images found in the current directory."
        )

        print(
            "  Add labeled/sample skin images to test "
            "actual application inference."
        )

    else:

        print(
            f"  Found {len(sample_images)} image(s)."
        )

        successful = 0

        for img_file in sample_images[:5]:

            try:

                prediction, confidence = (
                    predict_image(
                        img_file,
                        model,
                    )
                )

                print(
                    f"  {img_file[:30]:30} "
                    f"→ {prediction:12} "
                    f"({confidence:6.2f}%)"
                )

                successful += 1

            except Exception as e:

                print(
                    f"  {img_file[:30]:30} "
                    f"→ ERROR: {str(e)[:60]}"
                )

        if successful > 0:

            test_pass(
                f"{successful} real image(s) "
                "successfully processed."
            )

        else:

            test_fail(
                "No real sample images could be processed."
            )

except Exception as e:

    test_warning(
        f"Sample-image test could not complete: "
        f"{type(e).__name__}: {e}"
    )


# ============================================================
# TEST 12: PERFORMANCE BENCHMARK
# ============================================================

section(12, "Performance Benchmark")

try:

    # Recreate the test image in case a previous test changed it.
    random_array = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8,
    )

    Image.fromarray(
        random_array
    ).save(
        TEMP_IMAGE_PATH,
        format="JPEG",
        quality=95,
    )


    # Warm-up.
    predict_image(
        TEMP_IMAGE_PATH,
        model,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()


    times = []

    for _ in range(5):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        predict_image(
            TEMP_IMAGE_PATH,
            model,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)


    avg_time = float(
        np.mean(times)
    )

    min_time = min(times)
    max_time = max(times)

    throughput = (
        1 / avg_time
        if avg_time > 0
        else 0
    )


    print(
        f"  Average inference time: "
        f"{avg_time * 1000:.2f} ms"
    )

    print(
        f"  Minimum time: "
        f"{min_time * 1000:.2f} ms"
    )

    print(
        f"  Maximum time: "
        f"{max_time * 1000:.2f} ms"
    )

    print(
        f"  Throughput: "
        f"{throughput:.2f} predictions/second"
    )

    test_pass(
        "Performance benchmark completed."
    )

except Exception as e:

    test_warning(
        f"Performance benchmark failed: "
        f"{type(e).__name__}: {e}"
    )


# ============================================================
# CLEANUP
# ============================================================

try:

    if os.path.exists(
        TEMP_IMAGE_PATH
    ):
        os.remove(
            TEMP_IMAGE_PATH
        )

    print()
    print(
        "🧹 Temporary test files cleaned up."
    )

except Exception as e:

    test_warning(
        f"Could not remove temporary test file: {e}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("📊 FINAL TEST REPORT")
print("=" * 70)

print(
    f"  Passed tests:   {passed_tests}"
)

print(
    f"  Failed tests:   {failed_tests}"
)

print(
    f"  Warnings:       {warning_tests}"
)

print(
    f"  Model classes:  {len(CLASS_NAMES)}"
)

print(
    f"  Classes:        {', '.join(CLASS_NAMES)}"
)

print(
    f"  Device:         {device}"
)

print(
    f"  Architecture:   ResNet50"
)


# ============================================================
# DEPLOYMENT STATUS
# ============================================================

if failed_tests == 0:

    print()
    print("=" * 70)
    print("✅ TECHNICAL MODEL TESTS PASSED")
    print("=" * 70)

    print()
    print(
        "The model loads and performs inference successfully."
    )

    print(
        "This does NOT prove medical accuracy."
    )

    print()
    print(
        "For deployment, also test the model against a "
        "held-out labeled dataset."
    )

    print()
    print(
        "🚀 Streamlit command:"
    )

    print(
        "   streamlit run app.py"
    )

else:

    print()
    print("=" * 70)
    print("❌ MODEL TESTS REQUIRE ATTENTION")
    print("=" * 70)

    print()
    print(
        "Fix the failed tests before deploying the application."
    )

    sys.exit(1)
