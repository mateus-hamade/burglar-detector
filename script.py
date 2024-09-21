from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load model

trainresults = model.train(
    data='./data/data.yaml',
    epochs=25,
    batch=16,
    imgsz=640
)  # Train

valresults = model.val()

print(trainresults, valresults)  # Output training and validation results