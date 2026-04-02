import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MobileNet feature extractor ──────────────────────────────────────────────
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
feature_extractor = mobilenet.features.to(device)
feature_extractor.eval()

# ── LSTM classifier ───────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1280, 256,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.5)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

model = LSTMClassifier().to(device)
state_dict = torch.load("classifier/mobilenet.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ── preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_frames(video_path, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(frame))
    cap.release()
    return torch.stack(frames)   # (32, 3, 224, 224)

# ── views ─────────────────────────────────────────────────────────────────────
def index(request):
    return render(request, "classifier/index.html")

@csrf_exempt
def predict_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    video_file = request.FILES.get("video")
    if not video_file:
        return JsonResponse({"error": "No video uploaded"}, status=400)

    
    
    tmp_path = os.path.join(tempfile.gettempdir(), video_file.name)
    with open(tmp_path, "wb") as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    try:
        
        frames = extract_frames(tmp_path)          # (32, 3, 224, 224)
        frames = frames.unsqueeze(0).to(device)    # (1, 32, 3, 224, 224)
        B, T, C, H, W = frames.shape

        
        with torch.no_grad():
            flat = frames.view(B * T, C, H, W)
            feats = feature_extractor(flat)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(B, T, -1)           # (1, 32, 1280)

            
            outputs = model(feats)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = round(probs[pred].item() * 100, 2)

        label = "Shoplifter" if pred == 1 else "Non-Shoplifter"
        return JsonResponse({"prediction": label, "confidence": confidence})

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
