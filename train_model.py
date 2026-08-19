from ultralytics import YOLO

def main():
    # Load pretrained model
    model = YOLO("yolov8m.pt")   # You can change to yolov8s.pt or yolov8x.pt

    # Train the model
    model.train(
        data="data.yaml",   # Path to your YAML file
        epochs=50,
        imgsz=768,
        batch=16,
        device=0  # 0 for GPU, remove if CPU
    )

if __name__ == "__main__":
    main()