# PyTorch Begriffserklärungen

## 1. PyTorch
Ein leistungsstarkes und weit verbreitetes Open-Source-Framework für maschinelles Lernen und Deep Learning. Es wurde von Facebook AI entwickelt und bietet eine flexible, dynamische Umgebung für die Erstellung, das Training und die Optimierung von Modellen. PyTorch wird besonders für neuronale Netzwerke genutzt und unterstützt GPU-Beschleunigung für schnelle Berechnungen.

---

## 2. Tensor
Ein Tensor ist das zentrale Datenobjekt in PyTorch. Er ist eine verallgemeinerte Form von Vektoren und Matrizen und kann mehrdimensionale Daten speichern und verarbeiten. PyTorch-Tensoren sind ähnlich zu NumPy-Arrays, bieten jedoch die Möglichkeit, Berechnungen effizient auf GPUs durchzuführen.

### **Eigenschaften von Tensors:**
- Mehrdimensionale Datenstruktur
- Kann auf CPUs und GPUs verarbeitet werden
- Ermöglicht Autograd (automatische Differentiation)
- Bietet ähnliche Operationen wie NumPy


### **Codebeispiel:**
```python
import torch

# Erstellen eines Tensors
tensor = torch.tensor([1, 2, 3])
print(tensor)
```
---

## 3. Neuronales Netzwerk
Ein Modell im maschinellen Lernen, das von biologischen Nervensystemen inspiriert ist. Es besteht aus mehreren Schichten von "Neuronen", die durch Gewichte miteinander verbunden sind und trainiert werden, um komplexe Muster in Daten zu erkennen.

### **Bestandteile eines neuronalen Netzwerks:**
- **Eingabeschicht (Input Layer)**: Erhält die Rohdaten und leitet sie weiter.
- **Versteckte Schichten (Hidden Layers)**: Verarbeiten die Daten durch mathematische Operationen, um relevante Merkmale zu extrahieren.
- **Ausgabeschicht (Output Layer)**: Gibt das finale Ergebnis oder die Vorhersage aus.

---

## 4. Convolutional Neural Network (CNN)
Eine spezielle Art von neuronalen Netzwerken, die besonders für Bildverarbeitung optimiert ist. Sie bestehen aus mehreren Convolutional- und Pooling-Schichten, die automatisch wichtige Merkmale aus Bildern extrahieren.

### **Wichtige Bestandteile eines CNNs:**
- **Convolutional Layer**: Extrahiert Merkmale aus dem Bild mithilfe von Filtern.
- **Pooling Layer**: Reduziert die Dimensionalität der Daten und verbessert die Effizienz.
- **Fully Connected Layer**: Verbindet die extrahierten Merkmale mit der finalen Vorhersage.

### **Codebeispiel:**
```python
import torch
import torch.nn as nn

# Beispiel eines einfachen Convolutional Layers
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
input_image = torch.randn(1, 1, 5, 5)  # Beispielbild (Batchgröße: 1, Kanäle: 1, 5x5)
output_image = conv_layer(input_image)
print(output_image)

```
---

## 5. nn.Module
Eine Basisklasse für alle neuronalen Netzwerke in PyTorch. Durch Vererbung von `nn.Module` kann ein eigenes Modell definiert werden, das verschiedene Schichten und Berechnungen enthält.

### **Codebeispiel:**
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Fully connected Layer mit 10 Eingabewerten und 5 Ausgabewerten
    
    def forward(self, x):
        return self.fc(x)

# Modellinstanz erstellen und vorwärtsdurchlauf testen
model = SimpleModel()
input_data = torch.randn(1, 10)
output_data = model(input_data)
print(output_data)
```
---

## 6. Convolutional Layer
Eine Schicht innerhalb eines CNNs, die die Faltung (Convolution) auf Eingabedaten anwendet, um Merkmale zu extrahieren.

### **Wichtige Parameter:**
- **Filter (Kernels)**: Kleine Matrizen, die über das Bild gleiten, um Muster zu erkennen.
- **Stride**: Die Schrittweite, mit der der Filter über das Bild bewegt wird.
- **Padding**: Fügt Ränder hinzu, um die Eingabegröße beizubehalten.

---

## 7. ReLU (Rectified Linear Unit)
Eine Aktivierungsfunktion, die nicht-negative Werte unverändert lässt und negative Werte auf Null setzt. Sie verbessert die Trainingsgeschwindigkeit und verhindert das Verschwinden von Gradienten.


### **Codebeispiel:**
```python
import torch
import torch.nn as nn

# Beispiel für ReLU-Aktivierung
relu = nn.ReLU()
input_data = torch.tensor([-1.0, 0.0, 1.0])
output_data = relu(input_data)
print(output_data)
```
---

## 8. Max-Pooling
Eine Technik zur Reduktion der Dimensionalität eines Bildes, indem nur der größte Wert innerhalb eines bestimmten Bereichs (z. B. 2x2) behalten wird. Dies hilft dabei, relevante Merkmale zu erhalten und den Rechenaufwand zu reduzieren.

### **Codebeispiel:**
```python
import torch
import torch.nn as nn

# Max-Pooling Beispiel
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
input_image = torch.tensor([[[[1, 2], [3, 4]]]])
output_image = max_pool(input_image)
print(output_image)
```
---

## 9. forward-Methode
Eine Methode in `nn.Module`, die definiert, wie die Daten durch das Modell fließen. Hier werden die verschiedenen Schichten und Berechnungen des Netzwerks beschrieben.

---

## 10. torch.no_grad()
Ein Kontextmanager, der verhindert, dass PyTorch Gradienten speichert. Dies wird während der Inferenz (Vorhersage) genutzt, um Speicherplatz zu sparen und Berechnungen zu beschleunigen.

### **Codebeispiel:**
```python
import torch

# Beispiel für torch.no_grad()
with torch.no_grad():
    x = torch.randn(3, 3)
    y = x * 2
    print(y)
```
---

## 11. DataLoader
Ein PyTorch-Objekt, das die Daten in Batches aufteilt und effizient lädt. Besonders nützlich für das Training großer Datensätze, da es parallele Verarbeitung ermöglicht.

---

## 12. ImageFolder
Eine Klasse, die automatisch Bilder in Unterordnern als Kategorien interpretiert und Labels zuweist. Nützlich für die Verarbeitung von Bilddatensätzen, die in einer ordnerbasierten Struktur gespeichert sind.

---

## 13. optimizer
Ein Algorithmus zur Aktualisierung der Modellgewichte während des Trainings.

### **Häufig verwendete Optimierungsalgorithmen:**
- **SGD (Stochastic Gradient Descent)**: Standardmethode zur Gewichtsaktualisierung.
- **Adam**: Beliebter Optimierungsalgorithmus mit adaptiven Lernraten.

---

## 14. cross_entropy
Eine Verlustfunktion, die für Klassifikationsprobleme genutzt wird. Sie misst, wie stark die vorhergesagten Wahrscheinlichkeiten von den echten Labels abweichen.

---

## 15. torch.max
Findet den maximalen Wert in einem Tensor. Wird oft genutzt, um die wahrscheinlichste Klasse aus Modellvorhersagen zu extrahieren.

---

## 16. Batch
Eine Gruppe von Trainingsbeispielen, die gleichzeitig verarbeitet werden. Dies erhöht die Effizienz und Stabilität des Trainings.

---

## 17. train() und eval()
- **train()**: Aktiviert das Modell im Trainingsmodus, sodass Funktionen wie Dropout aktiviert werden.
- **eval()**: Schaltet das Modell in den Evaluierungsmodus, um Vorhersagen stabiler zu machen.

---

## 18. Epoch
Eine komplette Iteration über den gesamten Trainingsdatensatz. Mehrere Epochen sind oft notwendig, um das Modell optimal anzupassen.

---

## 19. Gradientenberechnung (Backpropagation)
Ein Verfahren zur Optimierung von Modellgewichten, bei dem der Fehler rückwärts durch das Netzwerk propagiert wird. Dabei werden Gradienten berechnet, die zur Anpassung der Gewichte verwendet werden.

---

## 20. Matplotlib
Eine Bibliothek zur Visualisierung von Daten in Form von Diagrammen und Grafiken. Sie wird oft verwendet, um Trainingsverläufe oder Bilddaten anzuzeigen.

---

## 21. make_grid
Eine Funktion aus `torchvision.utils`, die mehrere Bilder in einem Gitterlayout anordnet. Dies ist hilfreich für die Visualisierung von Batches.

---

## 22. state_dict()
Ein Python-Wörterbuch, das die aktuellen Modellparameter (Gewichte und Biases) speichert. Wird genutzt, um Modelle zu speichern und wieder zu laden.

---

## 23. Testen und Inferenz
- **Testen**: Bewertung des Modells mit einem separaten Testdatensatz.
- **Inferenz**: Nutzung des trainierten Modells, um Vorhersagen für neue, unbekannte Daten zu treffen.

---

