from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import io
from pathlib import Path
from typing import Dict

# --- Setup: App and Paths ---
app = FastAPI(title="AegisAI API")
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# --- Global Variables: Models, Classifiers, and Attack Objects ---
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load hardened PyTorch model
def load_hardened_model() -> nn.Module:
    model = resnet18(num_classes=10)
    model_path = MODEL_DIR / "hardened_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

hardened_model = load_hardened_model()
anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.pkl")

# Create ART classifier and attack object once on startup
art_classifier = PyTorchClassifier(
    model=hardened_model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(hardened_model.parameters()),
    input_shape=(3, 32, 32),
    nb_classes=10,
    device_type='gpu' if 'cuda' in DEVICE.type else 'cpu'
)

fgsm_attack = FastGradientMethod(estimator=art_classifier, eps=0.03)

# --- Pydantic Models for Data Validation ---
class AnomalyFeatures(BaseModel):
    impossible_travel_speed: float
    login_frequency_1hr: float
    ip_change_count_24hr: float

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    model_score: int

class RobustnessResponse(BaseModel):
    clean_prediction: str
    attacked_prediction: str

# --- Helper Functions ---
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def get_prediction(model: nn.Module, img_tensor: torch.Tensor) -> str:
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = outputs.max(1)
    return CIFAR10_CLASSES[pred.item()]

# --- API Endpoints ---
@app.post('/test_model_robustness', response_model=RobustnessResponse)
async def test_model_robustness(file: UploadFile = File(...)) -> Dict[str, str]: ## <-- THE FIX IS HERE
    image_bytes = await file.read()
    img_tensor = preprocess_image(image_bytes)

    # Generate adversarial image using the pre-built attack object
    adv_numpy = fgsm_attack.generate(img_tensor.cpu().numpy())
    adv_tensor = torch.from_numpy(adv_numpy).to(DEVICE)

    # Get predictions
    clean_pred = get_prediction(hardened_model, img_tensor)
    attacked_pred = get_prediction(hardened_model, adv_tensor)

    return {'clean_prediction': clean_pred, 'attacked_prediction': attacked_pred}

@app.post('/predict_anomaly', response_model=AnomalyResponse)
def predict_anomaly(features: AnomalyFeatures) -> Dict[str, any]:
    sample = np.array([list(features.dict().values())]).reshape(1, -1)
    score = int(anomaly_model.predict(sample)[0])
    is_anomaly = (score == -1)
    return {'is_anomaly': is_anomaly, 'model_score': score}

# --- Main Application Runner ---
if __name__ == "__main__":
    print(f"Starting FastAPI app on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 