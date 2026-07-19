from unittest.mock import MagicMock

class YOLO:
    """Mock YOLO class for test runtime."""
    def __init__(self, model_name="yolov8n.pt", *args, **kwargs):
        self.model_name = model_name
        self.device = "cpu"
        
    def to(self, device):
        self.device = device
        return self
        
    def __call__(self, frame, verbose=False, *args, **kwargs):
        # Returns a mock result structure matching ultralytics output
        mock_result = MagicMock()
        return [mock_result]
