import cv2
from detect import CowDetection
from jcopvision.io import MediaReader, key_pressed, create_sized_window
import config as cfg

if __name__ == "__main__":
    media = MediaReader(cfg.FILE_PATH)
    model = CowDetection(cfg.MODEL_PATH, cfg.CONFIG_PATH, cfg.LABEL_PATH)

    create_sized_window(500, media.aspect_ratio, cfg.WINDOW_NAME)

    for frame in media.read():
        # Perform Detection
        bbox, labels, scores = model.predict(frame[..., ::-1], min_confidence=cfg.MIN_CONF, max_iou=cfg.MAX_IOU)
        # Draw Bounding Box
        frame = model.draw(frame, bbox, labels, scores)
        # Visualize
        cv2.imshow(cfg.WINDOW_NAME, frame[..., ::-1])

        if key_pressed("q"):
            break
    media.close()
