import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt   

import preprocess as pr

import os
import time
import csv
import shutil

# Definiert eine CNN-Klassifikation für Bilddatensätze
class CNNClassification(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduziert die Höhe und Breite um die Hälfte

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduziert die Höhe und Breite um die Hälfte

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(256*13*17, 2048),  # Angepasste Dimension basierend auf Eingangsdaten
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128,2) #letzter schritt auf zwei vektoren umziehen 
        )

    def forward(self, xb):
        return self.network(xb)

    @torch.no_grad()  # Deaktiviert das Gradienten-Tracking (nützlich für Inferenz)
    def inferenzSet(self, dataset, device,logfile):

        self.eval()
        images = [sublist[0].to(device) for sublist in dataset]
        images = torch.stack(images).to(device)
        labels = [sublist[1] for sublist in dataset]
        labels = torch.tensor(labels).to(device)

        res = self(images)

        _, preds = torch.max(res, dim=1)

        accuracy = torch.sum(preds == labels).item() / len(preds)

        log_test_results(dataset,preds.cpu().tolist(),logfile)

        print("Erg: " + str(accuracy))

        return preds,accuracy

    def inferenzImages(self, device, dataset, start, length=1 ):
        with torch.no_grad():
            for i in range(start, start + length):
                img, label = dataset[i]
                img = img.to(device)
                res = self(img[None, :, :, :])  # Fügt eine Batch-Dimension hinzu
                _, pred = torch.max(res, dim=1)
                print(f"Index: {i} Predicted class: {pred[0].item()} Defined class: {label}")


    def trainStart(self, epochs, lr, train_loader, device,modelname, opt_func=torch.optim.Adam):
        optimizer = opt_func(self.parameters(), lr)
        self.to(device)  # Verschiebt das Modell auf das angegebene Gerät
        self.train()
        
        log_file= modelname[:-6]+".csv"
        print("log saved to:",log_file)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "epoch_loss", "epoch_acc", "timestamp"])

            for epoch in range(epochs):
                train_losses = []
                for batch in train_loader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    loss = self.training_step((images, labels))
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    self.eval()
                    outputs = [self.validation_step(batch, device) for batch in train_loader]
                    batch_losses, batch_accs = zip(*outputs)
                    epoch_loss = torch.stack(batch_losses).mean().item()
                    epoch_acc = torch.stack(batch_accs).mean().item()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"Epoch {epoch}, loss: {epoch_loss}, acc: {epoch_acc}, Timestamp: {timestamp}")
                    writer.writerow([epoch, epoch_loss, epoch_acc, timestamp])


    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return (loss.detach(), acc)

def log_test_results(test_dataset, predictions, filename="test_results.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Label", "Prediction"])

            for (img_path, label), pred in zip(test_dataset.samples, predictions):
                filename = os.path.basename(img_path)
                writer.writerow([filename, label, pred])
        
        print(f"Test results saved to {filename}")

def rgba_loader(path):
        from PIL import Image
        img = Image.open(path).convert("RGBA")  # Ensure RGBA mode
        return transforms.ToTensor()(img)  # Keep all 4 channels

def train_model(data_dir, device,epochs=5,modelname = "model.state"):
    trans = [transforms.ToTensor()]

    train_dataset = ImageFolder(data_dir, transform=None, loader=rgba_loader) 
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = CNNClassification()


    try:
        model.load_state_dict(torch.load(modelname))
    except:
        print("No model found")

    trainEpochs = epochs
    if trainEpochs > 0:
        model.trainStart(trainEpochs, 0.001, train_dl, device,modelname)
        torch.save(model.state_dict(), modelname)


def test_model(test_data_dir, device,modelname,logfile):
    trans = [transforms.ToTensor()]
     
    model = CNNClassification()
    test_dataset = ImageFolder(test_data_dir, transform=None,loader=rgba_loader)

    print(modelname)

    try:
        model.load_state_dict(torch.load(modelname))
        model.to(device)
    except:
        print("No model found")
        return

    preds,_ = model.inferenzSet(test_dataset, device,logfile)
    #model.inferenzImages(dataset=test_dataset, device=device,  start=0, length=len(test_dataset))


def get_device(preferred_device=None):
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def copy_misclassified_images(csv_file, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            filename, label, prediction = row
            if label != prediction:  # Copy only misclassified images
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)

def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Jibbet Nich")


if __name__ == '__main__':

    #pr.preprocess()

    data_dir = "Datensatz/Learn"
    test_data_dir = "Datensatz/Test"
     

    model = "model40_4Layer.state"

    logfile = model[:-6]+"_testlog.csv"

    fehl_data_dir = "Datensatz/"+model[:-6]+"fehl"

    # Geräteauswahl: "cuda", "mps" oder "cpu"
    preferred_device = "cuda"  # Beispiel: Manuelle Auswahl von MPS
    device = get_device(preferred_device)
    if not os.path.exists(data_dir):
        print(f"Fehler: Der Ordner {data_dir} existiert nicht!")

    print(f"Using device: {device}")

    train_model(data_dir, device, epochs=40,modelname=model)
    #train_model(data_dir, device, epochs=40,modelname="model40.state")
    #train_model(data_dir, device, epochs=60,modelname="model60.state")
    test_model(test_data_dir, device,model,logfile)

    #Sorge dafür, dass alle Bilder, bei denen es nicht geklappt hat, wegsortiert werden. 

    cleanup(fehl_data_dir)

    copy_misclassified_images(logfile,test_data_dir+"/maennlich",fehl_data_dir)
    copy_misclassified_images(logfile,test_data_dir+"/weiblich",fehl_data_dir)

