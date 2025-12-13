import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_dir = './data/U'
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Классы: {class_names}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    num_epochs = 20
    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")

    print(f"Обучение завершено за {(time.time() - start_time):.1f} сек.")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Точность на тесте: {accuracy:.2f}%")
    torch.save(model.state_dict(), 'ferrari_alexnet.pth')

    def imshow(inp, title=None):
        inp = inp.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)

    examples_per_class = 3
    selected_inputs = []
    selected_labels = []
    selected_preds = []
    class_counters = {cls: 0 for cls in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            _, preds = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                label = labels[i].item()
                if class_counters[label] < examples_per_class:
                    selected_inputs.append(inputs[i])
                    selected_labels.append(labels[i])
                    selected_preds.append(preds[i])
                    class_counters[label] += 1
                if all(count >= examples_per_class for count in class_counters.values()):
                    break
            if all(count >= examples_per_class for count in class_counters.values()):
                break

    for cls, count in class_counters.items():
        if count < examples_per_class:
            print(f"Предупреждение: Для класса {class_names[cls]} найдено только {count} примеров в тесте.")

    num_examples = len(selected_inputs)
    cols = 3
    rows = (num_examples + cols - 1) // cols
    plt.figure(figsize=(12, 4 * rows))
    for i in range(num_examples):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        imshow(selected_inputs[i])
        color = 'green' if selected_preds[i] == selected_labels[i] else 'red'
        plt.title(f'Пред: {class_names[selected_preds[i]]}\nФакт: {class_names[selected_labels[i]]}', color=color)

    plt.suptitle('Примеры предсказаний (по 3 на класс)', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()