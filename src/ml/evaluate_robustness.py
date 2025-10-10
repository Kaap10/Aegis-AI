import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod # <-- CORRECTED NAME
import numpy as np

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for evaluation.")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 test set
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Helper function for accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- Load Models ---
# Load Baseline Model
baseline_model = resnet18(weights=None, num_classes=10)
baseline_model.load_state_dict(torch.load("models/baseline_model.pt", map_location=device))
baseline_model.to(device)
baseline_model.eval()

# Load Hardened Model
hardened_model = resnet18(weights=None, num_classes=10)
hardened_model.load_state_dict(torch.load("models/hardened_model.pt", map_location=device))
hardened_model.to(device)
hardened_model.eval()

# --- Create ART Wrappers and Attack ---
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

# Wrapper for baseline model
baseline_classifier = PyTorchClassifier(
    model=baseline_model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(0, 1)
)

# Wrapper for hardened model
hardened_classifier = PyTorchClassifier(
    model=hardened_model,
    loss=loss_fn,
    optimizer=torch.optim.Adam(hardened_model.parameters(), lr=0.001),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(0, 1)
)

# Instantiate the attack with the corrected name
attack = FastGradientMethod(estimator=baseline_classifier, eps=0.15) # <-- CORRECTED NAME

# --- Generate Adversarial Test Set ---
print("Generating adversarial examples for the test set...")
x_test = np.concatenate([x.numpy() for x, y in test_loader])
y_test = np.concatenate([y.numpy() for x, y in test_loader])
x_test_adv = attack.generate(x=x_test)

# Convert adversarial examples back to a DataLoader
adv_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test_adv), torch.from_numpy(y_test))
adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=128, shuffle=False)

# --- Evaluate and Print Results ---
print("Calculating accuracies...")
acc_clean = calculate_accuracy(baseline_model, test_loader)
acc_attacked_baseline = calculate_accuracy(baseline_model, adv_loader)
acc_attacked_hardened = calculate_accuracy(hardened_model, adv_loader)

print("\n--- Model Robustness Evaluation ---")
print(f"Baseline Model Accuracy (Clean Data):    {acc_clean:.2f}%")
print(f"Baseline Model Accuracy (Attacked Data): {acc_attacked_baseline:.2f}%")
print(f"Hardened Model Accuracy (Attacked Data): {acc_attacked_hardened:.2f}%")
print("-----------------------------------")