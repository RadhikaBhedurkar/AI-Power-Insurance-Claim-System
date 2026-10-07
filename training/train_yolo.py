"""Step 2: damage detection. Convert CarDD (COCO json) -> YOLO labels first:
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir="data/cardd/annotations", use_segments=False)
Then create data/damage.yaml with train/val paths and class names
(CarDD: dent, scratch, crack, glass shatter, lamp broken, tire flat)."""
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="data/damage.yaml", epochs=60, imgsz=640, batch=16, project="models", name="yolo_damage")
# best weights -> models/yolo_damage/weights/best.pt  (copy to models/yolo_damage.pt)
