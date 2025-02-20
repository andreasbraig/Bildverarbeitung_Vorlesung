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
            # Erster Convolutional Layer (Eingabe: 3 Kanäle für RGB-Bilder, Ausgabe: 32 Kanäle)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # Aktivierungsfunktion ReLU nach dem Convolutional Layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Max-Pooling: Reduziert die Höhe und Breite um die Hälfte

            # Zweiter Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Nochmals Max-Pooling

            # Flatten und vollverbundene Schichten
            nn.Flatten(),  # Flacht die Ausgabe für die Übergabe an die Fully Connected Layer
            nn.Linear(-1, 128),  # Linear Layer (128 Neuronen, die vorherige Dimension wird automatisch berechnet)
            nn.ReLU(),  # Aktivierung für die vollverbundene Schicht
            nn.Linear(128, 2)  # Ausgangsschicht mit 2 Neuronen für Binärklassifikation
        )

    # Vorwärtsdurchlauf durch das Netzwerk
    def forward(self, xb):
        return self.network(xb)  # Gibt die Ausgabe des Netzwerks zurück
        
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
        #print("Res:",res)
        _, preds = torch.max(res, dim=1)  # Nimmt die Klasse mit der höchsten Wahrscheinlichkeit
        #print("Preds",preds)
        #print(len(labels))
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
                self.eval()  # Setzt das Modell in den Evaluierungsmodus
                outputs = [self.validation_step(batch) for batch in train_loader]
                batch_losses, batch_accs = zip(*outputs)
                epoch_loss = torch.stack(batch_losses).mean().item()  # Mittelwert des Losses über alle Batches
                epoch_acc = torch.stack(batch_accs).mean().item()  # Mittelwert der Genauigkeit über alle Batches
                print(f"Epoch {epoch}, loss: {epoch_loss}, acc: {epoch_acc}")

    def training_step(self, batch):
        # Führt einen Vorwärtsdurchlauf durch und berechnet den Cross-Entropy-Loss
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)  # Berechnet den Verlust mit Cross-Entropy für die Klassifikation
        return loss

    def validation_step(self, batch):
        # Führt die Inferenz durch und berechnet den Loss und die Genauigkeit
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)  # Berechnet den Verlust
        _, preds = torch.max(out, dim=1)  # Bestimmt die Vorhersageklasse
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))  # Berechnet die Genauigkeit
        return (loss.detach(), acc)

# Zeigt ein Bild aus dem Datensatz an
def showImage(dataSet, index):
    img, label = dataSet[index]
    print(f"Label : {dataSet.classes[label]}")  # Zeigt das Label des Bildes an
    print(img.shape, label)  # Zeigt die Form des Bildes und das Label an
    plt.imshow(img.permute(1, 2, 0))  # PyTorch-Bilder haben die Form (C, H, W), Matplotlib erwartet (H, W, C)
    plt.show()

# Zeigt eine Batch von Bildern an
def showBatch(dataset, index):
    batchImages = [images for images, _ in dataset]  # Extrahiert nur die Bilder
    images = batchImages[index]
    _, ax = plt.subplots(figsize=(16, 12))
    ax.set_xticks([])  # Entfernt die x-Achsen-Ticks
    ax.set_yticks([])  # Entfernt die y-Achsen-Ticks
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))  # Zeigt die Bilder in einem Raster an
    plt.show()

def train_model(data_dir):
        # Bildtransformationen: Größe ändern, in Graustufen umwandeln und in Tensor konvertieren
    trans = [
        #transforms.Resize((40, 40)),  # Bildgröße ändern (optional)
        #transforms.Grayscale(num_output_channels=1),  # In Graustufen umwandeln (optional)
        transforms.ToTensor()  # Wandelt das Bild in einen Tensor um
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Bestimmt das verwendete Gerät (MPS oder CPU)

    # Läd den Trainings- und Testdatensatz
    train_dataset = ImageFolder(data_dir, transform=transforms.Compose(trans))

    print(train_dataset)
    print("Classes in dataset: ", train_dataset.classes)  # Zeigt die Klassen im Datensatz an
  

    # DataLoader für das Training
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Modellinitialisierung
    model = CNNClassification()
    #model.to(device)  # Verschiebt das Modell auf das entsprechende Gerät (optional)

    try:
        model.load_state_dict(torch.load("model.state"))  # Läd das gespeicherte Modell
    except:
        print("No model found")  # Falls kein Modell gefunden wird

    # Training für 3 Epochen
    trainEpochs = 3
    if trainEpochs > 0:
        model.trainStart(trainEpochs, 0.001, train_dl)  # Startet das Training
        torch.save(model.state_dict(), "model.state")  # Speichert das Modell

def test_model(test_data_dir):

    trans = [
        #transforms.Resize((40, 40)),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]

    model = CNNClassification()
    test_dataset = ImageFolder(test_data_dir, transforms.Compose(trans))

    print("Classes in test dataset: ", test_dataset.classes)  # Zeigt die Klassen im Testdatensatz an

    try:
        model.load_state_dict(torch.load("model.state"))  # Läd das gespeicherte Modell
    except:
        print("No model found")  # Falls kein Modell gefunden wird
        return


    # Führt Inferenz auf dem Testdatensatz durch
    model.inferenzSet(test_dataset)
    model.inferenzImages(test_dataset, int(len(test_dataset) / 2), 2)  # Führt die Inferenz auf einer Teilmenge von Bildern durch


# Hauptprogramm
if __name__ == '__main__':
    # Datensatzpfade (lokaler Speicherort)
    data_dir = "../Bilder/Cashews/Lernen"
    test_data_dir = "../Bilder/Cashews/Test"

    print(torch.backends.mps.is_available())  # Überprüft, ob MPS verfügbar ist

    train_model(data_dir)  # Startet das Training

    #test_model(test_data_dir)  # Testet das Modell (falls erforderlich)
