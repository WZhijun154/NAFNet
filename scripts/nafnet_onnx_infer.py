import cv2
import numpy as np
import onnxruntime as ort


class NafNetInferPipeline:
    def __init__(self, onnx_path: str, print_model_format=False):
        self.onnx_path = onnx_path
        if print_model_format:
            self.get_onnx_model_input_format()
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(onnx_path, session_options)

    def get_onnx_model_input_format(self):
        """
        Prints the input format of an ONNX model.

        Parameters:
        model_path (str): Path to the ONNX model.
        """
        # Load the model
        import onnx

        model = onnx.load(self.onnx_path)

        # Get the model graph
        graph = model.graph

        # Iterate through the input nodes
        for input_node in graph.input:
            name = input_node.name
            shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value != 0 else "Unknown")
            dtype = input_node.type.tensor_type.elem_type

            print(f"Input Name: {name}")
            print(f"Shape: {shape}")
            print(f"Data Type: {onnx.TensorProto.DataType.Name(dtype)}\n")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # to float32
        img = img.astype(np.float32) / 255.0
        # add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def process_onnx(self, img: np.ndarray) -> np.ndarray:
        # run inference
        ort_inputs = {self.session.get_inputs()[0].name: img}
        output_name = self.session.get_outputs()[0].name
        ort_outs = self.session.run([output_name], ort_inputs)
        return ort_outs[0]

    def postprocess(self, img: np.ndarray) -> np.ndarray:
        # scale to 255
        img = (img * 255.0).clip(0, 255).round().astype(np.uint8)
        # squeeze batch dimension
        img = np.squeeze(img, axis=0)
        # CHW to HWC
        img = np.transpose(img, (1, 2, 0))
        # convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def __call__(self, img_path, output_path):
        img = cv2.imread(img_path)
        img = self.preprocess(img)
        img = self.process_onnx(img)
        img = self.postprocess(img)
        cv2.imwrite(output_path, img)


nafnet_sidd_infer_pipeline = NafNetInferPipeline(
    "experiments/onnx/nafnet_sidd_dynamic_simplified.onnx"
)

nafnet_sidd_infer_pipeline("./demo/noisy.png", "./demo/denoised.png")

nafnet_reds_infer_pipeline = NafNetInferPipeline(
    "experiments/onnx/nafnet_reds_dynamic_simplified.onnx"
)

nafnet_reds_infer_pipeline("demo/blurry.jpg", "demo/de-blurry-onnx.png")
