import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import os

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_baseline(model, trainloader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")
    return model

def train_hardened(model, trainloader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.03, eps_step=0.01, max_iter=20, batch_size=64)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Generate adversarial examples
            adv_inputs = torch.tensor(attack.generate(x=inputs.cpu().numpy()), device=device)
            # Combine clean and adversarial
            combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            optimizer.zero_grad()
            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Hardened] Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    device = get_device()
    trainloader, _ = get_data_loaders()
    # Baseline model
    baseline_model = get_resnet18()
    print("Training baseline model...")
    baseline_model = train_baseline(baseline_model, trainloader, device)
    save_model(baseline_model, os.path.join(os.path.dirname(__file__), '../../models/baseline_model.pt'))
    print("Baseline model trained and saved to models/baseline_model.pt")
    # Hardened model
    hardened_model = get_resnet18()
    print("Training hardened (adversarially trained) model...")
    hardened_model = train_hardened(hardened_model, trainloader, device)
    save_model(hardened_model, os.path.join(os.path.dirname(__file__), '../../models/hardened_model.pt'))
    print("Hardened model trained and saved to models/hardened_model.pt")

if __name__ == "__main__":
    main()
