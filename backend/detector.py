import os
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = [
    'Grape__BlackRot', 
    'Grape__Esca', 
    'Grape__Healthy', 
    'Grape__LeafBlight'
]

COLORS = [
    (255, 0, 0),     # Blue - BlackRot
    (255, 255, 0),   # Cyan - Esca
    (0, 255, 0),     # Green - Healthy
    (0, 165, 255)    # Orange - LeafBlight
]

class GrapeDiseaseDetector:
    def __init__(self, model_path: str = "model/best.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[DETECTOR] Warning: Model file not found at {self.model_path}")
            return
        
        # Use CPU provider or CUDA if configured in container
        self.session = ort.InferenceSession(
            self.model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[DETECTOR] ONNX session loaded successfully from {self.model_path}")

    #Resize and pad image while preserving aspect ratio
    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    #Runs inference on an image and returns results + annotated image
    def predict(self, image_path: str, conf_threshold: float = 0.55, iou_threshold: float = 0.45):
        if self.session is None:
            raise RuntimeError("Model is not initialized.")

        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        img, ratio, (pad_w, pad_h) = self.letterbox(orig_img, new_shape=(640, 640))
        
        # Preprocessing: BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = blob.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        predictions = np.squeeze(outputs).T

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return orig_img, []

        # Convert cxcywh to x1y1x2y2
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Scale back to original resolution
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio

        h_orig, w_orig = orig_img.shape[:2]
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)

        # NMS
        max_wh = 4096.0
        c = class_ids * max_wh
        x1_offset = x1 + c
        y1_offset = y1 + c

        boxes_for_nms = np.stack([x1_offset, y1_offset, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences.tolist(), conf_threshold, iou_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
                score = float(confidences[i])
                cls_id = int(class_ids[i])
                cls_name = CLASS_NAMES[cls_id]
                color = COLORS[cls_id]

                # Draw bounding box on original image
                cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{cls_name}: {score:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(orig_img, (box[0], box[1] - text_h - 10), (box[0] + text_w, box[1]), color, -1)
                cv2.putText(orig_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                results.append({
                    "disease_name": cls_name,
                    "confidence": round(score, 3),
                    "bounding_box": box
                })

        return orig_img, results    