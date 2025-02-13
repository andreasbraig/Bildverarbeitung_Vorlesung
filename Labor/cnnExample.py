import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt    

# Definiert eine CNN-Klassifikation für Bilddatensätze
class CNNClassification(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Das Netzwerk besteht aus mehreren Convolutional Layers, ReLU-Aktivierungen und Max-Pooling-Schichten
        self.network = nn.Sequential(
            # Erster Convolutional Layer (Eingabe: 1 Kanal, Ausgabe: 32 Kanäle)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduziert die Höhe und Breite um die Hälfte

            # Zweiter Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Dritter Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Flatten und vollverbundene Schichten
            nn.Flatten(),
            nn.Linear(6400, 128),  # 6400 = berechnete Größe nach Max-Pooling
            nn.ReLU(),
            nn.Linear(128, 2)  # Zwei Klassen (Binärklassifikation)
        )

    # Vorwärtsdurchlauf durch das Netzwerk
    def forward(self, xb):
        return self.network(xb)
        
    @torch.no_grad()  # Deaktiviert das Gradienten-Tracking (nützlich für Inferenz)
    def inferenzSet(self, dataset):
        self.eval()  # Setzt das Modell in den Evaluierungsmodus (kein Dropout etc.)
        
        # Extrahiert alle Bilder und Labels aus dem Datensatz
        images = [sublist[0] for sublist in dataset]
        images = torch.stack(images)  # Stapelt die Bilder zu einem Tensor

        labels = [sublist[1] for sublist in dataset]
        labels = torch.tensor(labels)

        # Führt die Vorhersage auf den gesamten Datensatz durch
        res = self(images)
        _, preds = torch.max(res, dim=1)  # Nimmt die Klasse mit der höchsten Wahrscheinlichkeit
        print("Erg: " + str(torch.sum(preds == labels).item() / len(preds)))  # Gibt die Genauigkeit aus

    def inferenzImages(self, dataset, start, length=1):
        # Führt die Inferenz auf einer Auswahl von Bildern durch
        with torch.no_grad():
            for i in range(start, start + length):
                img, label = dataset[i]
                res = self(img[None, :, :, :])  # Fügt eine Batch-Dimension hinzu
                _, pred = torch.max(res, dim=1)  # Nimmt die Klasse mit der höchsten Wahrscheinlichkeit
                print(f"Index: {i} Predicted class: {pred[0].item()} Defined class: {label}")

    def trainStart(self, epochs, lr, train_loader, opt_func=torch.optim.Adam):
        # Initialisiert den Optimierer
        optimizer = opt_func(self.parameters(), lr)
        self.train()  # Setzt das Modell in den Trainingsmodus
        
        for epoch in range(epochs):
            train_losses = []
            
            # Batchweise Training
            for batch in train_loader:
                loss = self.training_step(batch)  # Berechnet den Loss
                train_losses.append(loss)
                loss.backward()  # Backpropagation
                optimizer.step()  # Aktualisiert die Modellparameter
                optimizer.zero_grad()  # Setzt die Gradienten auf Null

            # Nach jedem Epoch: Evaluierung der Trainingsgenauigkeit
            with torch.no_grad():
                self.eval()
                outputs = [self.validation_step(batch) for batch in train_loader]
                batch_losses, batch_accs = zip(*outputs)
                epoch_loss = torch.stack(batch_losses).mean().item()
                epoch_acc = torch.stack(batch_accs).mean().item()
                print(f"Epoch {epoch}, loss: {epoch_loss}, acc: {epoch_acc}")

    def training_step(self, batch):
        # Führt einen Vorwärtsdurchlauf durch und berechnet den Cross-Entropy-Loss
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        # Führt die Inferenz durch und berechnet den Loss und die Genauigkeit
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return (loss.detach(), acc)

# Zeigt ein Bild aus dem Datensatz an
def showImage(dataSet, index):
    img, label = dataSet[index]
    print(f"Label : {dataSet.classes[label]}")
    print(img.shape, label)
    plt.imshow(img.permute(1, 2, 0))  # PyTorch-Bilder haben die Form (C, H, W), Matplotlib erwartet (H, W, C)
    plt.show()

# Zeigt eine Batch von Bildern an
def showBatch(dataset, index):
    batchImages = [images for images, _ in dataset]
    images = batchImages[index]
    _, ax = plt.subplots(figsize=(16, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()

# Hauptprogramm
if __name__ == '__main__':
    # Datensatzpfade (lokaler Speicherort)
    data_dir = "../ZhangLabData/train"
    test_data_dir = "../ZhangLabData/test"

    # Bildtransformationen: Größe ändern, in Graustufen umwandeln und in Tensor konvertieren
    trans = [
        transforms.Resize((40, 40)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]

    # Läd den Trainings- und Testdatensatz
    train_dataset = ImageFolder(data_dir, transform=transforms.Compose(trans))
    test_dataset = ImageFolder(test_data_dir, transforms.Compose(trans))

    print(train_dataset)
    print("Classes in dataset: ", train_dataset.classes)
    print("Classes in test dataset: ", test_dataset.classes)

    # DataLoader für das Training
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Modellinitialisierung
    model = CNNClassification()
    try:
        model.load_state_dict(torch.load("model.state"))  # Läd das gespeicherte Modell
    except:
        print("No model found")

    # Training für 3 Epochen
    trainEpochs = 3
    if trainEpochs > 0:
        model.trainStart(trainEpochs, 0.001, train_dl)
        torch.save(model.state_dict(), "model.state")  # Speichert das Modell

    # Führt Inferenz auf dem Testdatensatz durch
    model.inferenzSet(test_dataset)
    model.inferenzImages(test_dataset, int(len(test_dataset) / 2), 2)
