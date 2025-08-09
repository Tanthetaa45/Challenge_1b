import cv2
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10("model.pt")

# Perform prediction
det_res = model.predict(
    "test4.jpg",   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold# Device to use (e.g., 'cuda:0' or 'cpu')
)

# Annotate and save the result
annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite("result1.jpg", annotated_frame)