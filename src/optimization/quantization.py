from pathlib import Path


def apply_int8_quantization(model_path: str, output_dir: str) -> Path:
    """Placeholder INT8 quantization hook.

    Replace with bitsandbytes/ONNX/TensorRT-LLM flow depending on your stack.
    """

    src = Path(model_path)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    marker = dst / "quantization_stub.txt"
    marker.write_text(
        f"TODO: quantize model from {src} to INT8 artifacts.\n",
        encoding="utf-8",
    )
    return marker
