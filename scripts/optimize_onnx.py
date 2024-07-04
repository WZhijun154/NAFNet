import subprocess

import onnx
import onnxoptimizer
from onnxruntime.quantization import QuantType, quantize_dynamic


def optimize_and_simplify_onnx_model(
    input_model_path, optimized_model_path, simplified_model_path
):
    """
    Optimizes and simplifies an ONNX model.

    Parameters:
    input_model_path (str): Path to the input ONNX model.
    optimized_model_path (str): Path to save the optimized ONNX model.
    simplified_model_path (str): Path to save the simplified ONNX model.
    """
    # Load the original model
    model = onnx.load(input_model_path)

    # Apply ONNX Optimizer
    passes = ["eliminate_deadend", "fuse_bn_into_conv", "eliminate_nop_dropout"]
    optimized_model = onnxoptimizer.optimize(model, passes)
    onnx.save(optimized_model, optimized_model_path)

    # Simplify the optimized model
    subprocess.run(
        ["python3", "-m", "onnxsim", optimized_model_path, simplified_model_path]
    )

    print("Model optimization and simplification complete.")


def quantize_onnx_model(simplified_model_path, infer_model_path, quantized_model_path):
    """
    Applies dynamic quantization to an ONNX model.

    Parameters:
    simplified_model_path (str): Path to the simplified ONNX model.
    quantized_model_path (str): Path to save the quantized ONNX model.
    """
    # preprocess
    # run example python -m onnxruntime.quantization.preprocess --input mobilenetv2-7.onnx --output mobilenetv2-7-infer.onnx
    subprocess.run(
        [
            "python3",
            "-m",
            "onnxruntime.quantization.preprocess",
            "--input",
            f"{simplified_model_path}",
            "--output",
            f"{infer_model_path}",
        ]
    )

    # Apply dynamic quantization
    quantize_dynamic(
        infer_model_path, quantized_model_path, weight_type=QuantType.QUInt8
    )

    print("Model quantization complete.")


def optimzie_nafnet_sidd_dynamic():
    onnx_path = "experiments/onnx/nafnet_sidd_dynamic.onnx"
    optimized_path = "experiments/onnx/nafnet_sidd_dynamic_optimized.onnx"
    simplified_path = "experiments/onnx/nafnet_sidd_dynamic_simplified.onnx"
    enable_dynamic_quantization = True
    optimize_and_simplify_onnx_model(onnx_path, optimized_path, simplified_path)
    if enable_dynamic_quantization:
        quantized_path = "experiments/onnx/nafnet_sidd_dynamic_quantized.onnx"
        infer_model_path = "experiments/onnx/nafnet_sidd_dynamic_infer.onnx"
        quantize_onnx_model(simplified_path, infer_model_path, quantized_path)


def optimzie_nafnet_reds_dynamic():
    onnx_path = "experiments/onnx/nafnet_reds_dynamic.onnx"
    optimized_path = "experiments/onnx/nafnet_reds_dynamic_optimized.onnx"
    simplified_path = "experiments/onnx/nafnet_reds_dynamic_simplified.onnx"
    enable_dynamic_quantization = True
    optimize_and_simplify_onnx_model(onnx_path, optimized_path, simplified_path)
    if enable_dynamic_quantization:
        quantized_path = "experiments/onnx/nafnet_reds_dynamic_quantized.onnx"
        infer_model_path = "experiments/onnx/nafnet_reds_dynamic_infer.onnx"
        quantize_onnx_model(simplified_path, infer_model_path, quantized_path)


def optimzie_tiny():
    onnx_path = "onnx/ddcolor_tiny.onnx"
    optimized_path = "onnx/ddcolor_tiny_optimized.onnx"
    simplified_path = "onnx/ddcolor_tiny_simplified.onnx"
    enable_dynamic_quantization = True
    optimize_and_simplify_onnx_model(onnx_path, optimized_path, simplified_path)
    if enable_dynamic_quantization:
        quantized_path = "onnx/ddcolor_tiny_quantized.onnx"
        infer_model_path = "onnx/ddcolor_tiny_infer.onnx"
        quantize_onnx_model(simplified_path, infer_model_path, quantized_path)


if __name__ == "__main__":
    # optimzie_large()
    # optimzie_nafnet_sidd_dynamic()
    optimzie_nafnet_reds_dynamic()
