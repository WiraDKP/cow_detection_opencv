import cv2
import numpy as np
from jcopvision.utils.bbox import denormalize_bbox
from jcopvision.draw import draw_single_bbox
from jcopvision.utils import non_max_suppression


class CowDetection:
    def __init__(self, model_pb, config_pbtxt, coco_label_txt):
        self.net = cv2.dnn.readNetFromTensorflow(model_pb, config_pbtxt)
        self.labels = self._read_label(coco_label_txt)
        
    def predict(self, image, min_confidence=0.3, max_iou=0.3):
        # Feedforward
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # image_id, label, conf, x1, y1, x2, y2
        result = np.array([det[1:] for det in detections[0, 0] if det[2] > min_confidence])
        label_ids = result[:, 0].astype(int)
        scores = result[:, 1]
        boxes = np.clip(result[:, 2:], 0, 1)

        # NMS
        idxs = non_max_suppression(boxes, scores, max_iou)
        boxes = boxes[idxs]
        labels = self.labels[label_ids[idxs]]
        scores = scores[idxs]
        return boxes, labels, scores
    
    def draw(self, frame, bbox, labels, scores):
        bbox = denormalize_bbox(frame, bbox)
        for box, label, conf in zip(bbox, labels, scores):
            frame = draw_single_bbox(frame, box, "yellow", label.title(), conf, thickness=2)
        return frame
    
    @staticmethod
    def _read_label(label_txt_file):
        with open(label_txt_file, "r") as f:
            labels = [line.strip("\n") for line in f]
        return np.array(labels)
