
from ultralytics import YOLO
import subprocess

def finetune():
    model = YOLO("yolov8l.yaml")
    model.train(data = 'datasets.yaml', epochs=120,patience=10, imgsz=864, batch=4)

    command = [
            "yolo", "val",   
            "--model", "/runs/detect/large/weights/best.pt",
            "--data", "datasets.yaml",
            "--imgsz", "864",
            "--batch", "1",
        ]
        try:
            print("start training with command:", " ".join(command))
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Output:", result.stdout)
            print("Error:", result.stderr)
            print("finished with return code:", result.returncode)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code:", e.returncode)
            print("Error:", e.stderr)
        # Execute the command
