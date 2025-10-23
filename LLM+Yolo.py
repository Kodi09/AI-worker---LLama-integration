import cv2
import torch
from pathlib import Path
import sys
import time
import requests  # ✅ replaced OpenAI client with Ollama local requests

# ========== LLaMA via Ollama ==========
def decide_robot(object_label):
    prompt = f"""
    You control 3 robotic arms:
    - Robot 1: METAL
    - Robot 2: PLASTIC
    - Robot 3: PAPER
    The camera detected: "{object_label}".
    Which robot should pick it? Reply only: Robot 1, Robot 2, or Robot 3.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",   # or "llama3.1" if you have it
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        data = response.json()
        decision = data.get("response", "").strip()
    except Exception as e:
        decision = f"Error: {e}"

    print(f"🤖 LLaMA Decision: {decision}")
    return decision

# ========== YOLOv7 Setup ==========
FILE = Path(__file__).resolve()
ROOT = FILE.parent / "yolov7"
sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

weights = 'best.pt'
device = select_device('cpu')
model = attempt_load(weights, map_location=device)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

class_colors = {0: (0,255,0), 1: (255,0,0), 2: (0,0,255)}

# ========== Camera ==========
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(1)  # short delay for camera to stabilize

print("Camera started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls_id = int(cls)
                label = f'{names[cls_id]} {conf:.2f}'
                color = class_colors.get(cls_id, (255, 255, 0))
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 🔹 Call LLaMA once per detection
                object_name = names[cls_id]
                decide_robot(object_name)

    cv2.imshow("YOLOv7-Tiny Detection + LLaMA", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
