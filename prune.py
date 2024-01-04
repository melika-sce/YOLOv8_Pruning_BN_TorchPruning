from ultralytics import YOLO
# load a pretrained model
model = YOLO('N:/Master/Thesis/Code/HPC/YOLO/yolov8/yolov8n_100_vis_1024_8/weights/best.pt')
results = model.train(data='VisDrone.yaml', epochs=1, imgsz=1024, batch=2, device=[0], name='yolov8n_50%prune', prune=True, prune_ratio=0.0010, prune_iterative_steps=1)
