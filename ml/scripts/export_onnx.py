import argparse
import os
from ultralytics import YOLO


def export_model_to_onnx(
    weights_path: str = "best.pt",
    imgsz: int = 640,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 12,
    half: bool = False,
):
    """
    Exports YOLOv8 model weights from PyTorch (.pt) format to ONNX (.onnx) format.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file '{weights_path}' was not found.")

    print(f"📦 Loading PyTorch model from '{weights_path}'...")
    model = YOLO(weights_path)

    print(
        f"⚙️  Starting ONNX export (imgsz={imgsz}, dynamic={dynamic}, simplify={simplify}, opset={opset})..."
    )

    # Run Ultralytics built-in exporter
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        half=half,
    )

    print(f"✅ Export completed successfully. File saved to: '{exported_path}'")
    return exported_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert YOLOv8 weights from PyTorch (.pt) to ONNX (.onnx) format."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to the input PyTorch weights (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size and dynamic image resolution",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="Disable graph simplification via onnxslim / onnx-simplifier",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX Opset version (default: 12)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 (half-precision) format",
    )

    args = parser.parse_args()

    export_model_to_onnx(
        weights_path=args.weights,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        half=args.half,
    )