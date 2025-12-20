import torch
import numpy as np
from ultralytics import YOLO as yl
from config import MODEL_PATH, CONFIDENCE

class CrashDetector:
    def __init__(self):
        print("Loading yolo model...")
        try:
            self.model = yl(MODEL_PATH)  # 'rtdetr-x.pt' - auto-handles download
        except Exception as e:
            print(f"Download failed: {e}. Run: pip install --upgrade ultralytics")
            print(f"Then: yolo predict model={MODEL_PATH}")
            raise
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RT-DETR-X loaded on {self.model.device} - Ready for crash detection!")
    
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return frame
        # Real-time optimized inference for RT-DETR
        results = self.model(frame, verbose=False, conf=CONFIDENCE, device=self.model.device)
        return results[0].plot()
