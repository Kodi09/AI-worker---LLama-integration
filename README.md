# 🧠 AI Worker — LLaMA Integration

This repository contains the integration of a **LLaMA-based Large Language Model (LLM)** with a **YOLOv7 object detection** pipeline for real-time decision making.  
The system allows the LLM to interpret object detection results (e.g., “Camera detected: plastic bottle”) and map them to task-specific robot actions — such as assigning the correct robotic arm for sorting.

---

## ⚙️ Installation

1️⃣ Clone this repository 
```bash
git clone https://github.com/Kodi09/AI-worker---LLama-integration.git  
cd AI-worker---LLama-integration  
```
---

2️⃣ Install dependencies  
```bash
pip install -r requirements.txt  
```
Download YOLOv7 (if not already included):  
```bash
git clone https://github.com/WongKinYiu/yolov7.git  
```
---
3️⃣ Install Ollama

Ollama lets you run LLaMA models locally (no internet or API key needed).
Download it from the official site for your platform: 🔗 https://ollama.ai/download.
After installing, verify it’s working:
```bash
ollama --version
```
---
4️⃣ Start Ollama (for LLaMA)  
```bash
ollama serve  
ollama run llama3  
```
---

5️⃣ Run the integration  
```bash
python LLM+Yolo.py  
```
---

## 🧠 Example Workflow
Camera → YOLOv7 Detection → LLaMA Reasoning → Robot Arm Assignment  

1. YOLO detects: plastic bottle  
2. LLaMA processes text prompt: “Camera detected: plastic bottle”  
3. LLM returns decision: “Assign to Robot Arm 2 (plastic)”  
4. Robot control module executes the corresponding movement.   

---

## 🙌 Credits
YOLOv7 by WongKinYiu  
Project developed at NYU Robotics Lab as part of AI agent autonomous sorting research.  
