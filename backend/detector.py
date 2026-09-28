import cv2
import numpy as np
import onnxruntime as ort

# Dataset class names
CLASS_NAMES = [
    'Grape__BlackRot', 
    'Grape__Esca', 
    'Grape__Healthy', 
    'Grape__LeafBlight'
]

# Bounding box colors (BGR format)
COLORS = [
    (255, 0, 0),     # Blue - BlackRot
    (255, 255, 0),   # Cyan - Esca
    (0, 255, 0),     # Green - Healthy
    (0, 165, 255)    # Orange - LeafBlight
]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and pads image while preserving aspect ratio (YOLOv8 letterboxing).
    """
    shape = img.shape[:2]  # Current height and width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2  # Divide padding equally on both sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


class GrapeDiseaseDetector:
    """
    ONNX Runtime inference wrapper for YOLOv8 Grape Disease Detection.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.55, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 1. Initialize ONNX Runtime Session
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_path: str):
        """
        Runs object detection on a given image file.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            tuple: (annotated_image, list_of_detection_dicts)
        """
        # 2. Load and preprocess image
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Failed to load image at: {image_path}")

        img, ratio, (pad_w, pad_h) = letterbox(orig_img, new_shape=(640, 640))
        
        # BGR to RGB, HWC to CHW, normalize to [0.0, 1.0]
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = blob.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # Shape: [1, 3, 640, 640]

        # 3. Perform inference via ONNX Runtime
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]  # Shape: [1, 8, 8400]

        # 4. Post-processing (YOLOv8 output format: [batch, 4 + num_classes, anchors])
        predictions = np.squeeze(outputs).T  # Transpose to [8400, 8]

        # Extract bounding boxes (cx, cy, w, h) and class confidence scores
        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Get class with highest confidence score per box
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Apply confidence threshold mask
        mask = confidences >= self.conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            print("No objects detected above the confidence threshold.")
            return orig_img, []

        # Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Rescale bounding boxes to original image resolution
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio

        # Clip boxes to original image dimensions
        h_orig, w_orig = orig_img.shape[:2]
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)

        # Apply class-aware coordinate offset for NMS
        max_wh = 4096.0
        c = class_ids * max_wh
        x1_offset = x1 + c
        y1_offset = y1 + c

        # 5. Class-Aware Non-Maximum Suppression (NMS)
        boxes_for_nms = np.stack([x1_offset, y1_offset, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, 
            confidences.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )

        results = []
        # 6. Draw bounding boxes and labels
        if len(indices) > 0:
            for i in indices.flatten():
                box = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
                score = float(confidences[i])
                cls_id = int(class_ids[i])
                cls_name = CLASS_NAMES[cls_id]
                color = COLORS[cls_id]

                # Draw bounding box
                cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Draw label text background and text
                label = f"{cls_name}: {score:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(orig_img, (box[0], box[1] - text_h - 10), (box[0] + text_w, box[1]), color, -1)
                cv2.putText(orig_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                results.append({
                    "class_name": cls_name,
                    "confidence": score,
                    "bbox": box
                })

        return orig_img, results


if __name__ == "__main__":
    MODEL_PATH = "best.onnx"          # Path to ONNX weights file
    IMAGE_PATH = "test_leaf.jpg"      # Path to input test image
    OUTPUT_PATH = "result_onnx.jpg"   # Path for output annotated image

    # Initialize the detector
    detector = GrapeDiseaseDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.55,
        iou_threshold=0.45
    )

    # Run disease detection
    annotated_img, detections = detector.predict(image_path=IMAGE_PATH)

    # Save output image
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    print(f"Result saved to {OUTPUT_PATH}")
    print("Detections:", detections)