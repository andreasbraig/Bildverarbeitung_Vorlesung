# Doku

## Gleiderung
* Probelemstellung
* Stand der Technik
    * Python
    * OpenCV
    * PyTorch
    * was ist möglich (big data, deeplearning, ...)
* Datensatz
    * Herausforderungen
    * Allgemein
* Code
    * Preprocessing
    * Segmentierung
    * CNN Modell
* Lösungsansätze im Vergeleich
    * Training und Evaluation
* Fazit

## Problemstellung
Diese Arbeit befasst sich Mit der Verarbeitung und Klassifikation einzelner Datenpunkte in einem Datensatz.
Die hierbei verwendeten Daten sind Gesichtsaufnahmen verschiedener Personen verschiedenen Alters. 

Die erste Teilaufgabe besteht in der Segmentierung und Verarbeitung dieses bereitgestellten Datensatzes, um die Gesichtselemente einheitlich im Bild zu positionieren. 
Augen und Mund sollen hierbei immer ein gleichschenkliges Dreieck an einer festen Position im Bild bilden. 

Die zweite Teilaufgabe befasst sich mit dem Klassifikationsproblem. 
Der verarbeitete Datensatz wird in ein Convolutional Neural Network geladen und das Geschlecht der abgebildeten Person klassifiziert. 
Hierbei soll das Netz eine binäre Klassifikation zwischen Männlich (0) und Weiblich (1) durchführen.

Für die Lösung dieser Aufgabe wurde die Programmiersprache Python mit den wesentlichen Bibliotheken "OpenCV", "numpy" und "Pytorch" verwendet. 


## Stand der Technik
### Python: 
Python ist eine interpretierte Datenorientierte Programmiersorache, mit welcher in kürzester Zeit mächtige Prorammabläufe durchgeführt werden können. 
### OpenCV:
OpenCV ist eine Biblioethek für Python, die es ermöglicht Bilder zu laden, zu verändern und wieder abzuspeichern. Vorteile dieser Bibliothek sind die bereits implementierten Funktionen.
### PyTorch: 

### Möglichkeiten:


## Datensatz
Hier gehts um die Bilder lol.
### Allgemein
Der Datensatz für dieses Projekt besteht aus den Bildern und den dazugehörigen Masken, sowie der tag.json Datei. Auf dieser JSON-Datei sind über die Dateinamen jeweils das Geschlecht der abgebildeten Person zugeordnet. Hiermit lässt  sich also der Datensatz in die Kategorien "männlich" und "weiblich" einteilen.
### Herausforderungen
Das Verarbeiten der Bilder auf einen gemeinsamen Nenner bietet eine gewisse Herausforderung. Hier ist jeweils aus der Maske die Information für die Position der Augen und des Mundes zu bestimmen. Mithilfe dieser Informationen werden die Bilder auf eine gemeinsame Bildgröße Verkleinert und die Augen und der Mund auf die selbe Position gebracht. Dadurch wird gewährleistet, dass das neuronale Netzwerk diese drei Punkte als Anhaltspunkte für den Vergleich und das Einlernen der Unterscheidungsmerkmale nutzen kann. 


## Code
Hier geht's um den eigentlichen Code (ohne Code-Snippets)
### Segmentierung
Die Datei "preprocess.py" dient der Verarbeitung des Datensatzes. Hier werden gezielt Funktionen implementiert, um die gegebenen Ressourcen (Segmentierungsmaske) zu nutzen und die Personen auf den Bildern aufgrund dessen Freizustellen und in Position zu transformieren. 

Für die Transformation wird die Affine Transformation aus der Bibliothek "OpenCV" verwendet. Diese bewirkt, dass gewisse Mekrmale im Bild erhalten bleiben, wie zum Beispiel die Parallelität von Elementen im Bild. Hierzu werden die Scherpunktpositionen der Konturen von Augen und Mund verwendet. 

Der Code in "segmentierung.py" führt verschiedene Bildverarbeitungsaufgaben durch, um die Person im Vordergrund zu isolieren und auf eine gemeinsame Zielmatrix zu transformieren. Hierzu sind einige Funktionen implementiert.

Mit der Funktion "freistellen" erwartet die Parameter "image" und "segm". "image" ist ein Bild im BGR-Format und "segm" die dazugehörige Segmentierungsmaske im selbigen Format. Diese Maske wird dazu verwendet, um die Person im Bild isolieren und den Hintergrund transparent bzw. mit schwarzem Alpha Wert einzufärben. Ausgabe dieser Funktion ist ein Bild im BGRA-Format. 


- `transformation(image, segm, eye_dist=70)`: Transformiert das Bild, um Augen und Mund auf eine feste Position zu bringen.
- `get_corners(segm)`: Bestimmt die Positionen der Augen und des Mundes aus der Maske.
- `get_single_center(segm, value)`: Findet das Zentrum eines bestimmten Segments im Bild.
- `get_lastpoint(centr)`: Berechnet den letzten Punkt basierend auf den Positionen der Augen.

Diese Funktionen helfen dabei, die Bilder auf eine einheitliche Größe zu bringen und die relevanten Gesichtspunkte zu normalisieren, um sie für ein neuronales Netzwerk vorzubereiten.

### Preprocessing

### CNN Modell
Der Code in `classification_HW.py` implementiert ein Convolutional Neural Network (CNN) zur Klassifikation von Gesichtsaufnahmen. Die Hauptkomponenten und Funktionen umfassen:

- `CNNClassification`: Eine Klasse, die das CNN-Modell definiert. Das Netzwerk besteht aus mehreren Convolutional Layers, ReLU-Aktivierungen und MaxPooling Layers, gefolgt von Fully Connected Layers.
- `forward(xb)`: Führt einen Vorwärtsdurchlauf durch das Netzwerk durch.
- `inferenzSet(dataset, device, logfile)`: Führt die Inferenz auf einem Datensatz durch und speichert die Ergebnisse in einer Logdatei.
- `trainStart(epochs, lr, train_loader, device, modelname, opt_func, patience, lr_patience, lr_decay_factor)`: Startet das Training des Modells mit Early Stopping und Learning Rate Decay.
- `training_step(batch)`: Berechnet den Verlust für einen Trainingsbatch.
- `validation_step(batch, device)`: Berechnet den Verlust und die Genauigkeit für einen Validierungsbatch.
- `apply_augmentation(image)`: Wendet Datenaugmentation auf ein Bild an.
- `log_test_results(test_dataset, predictions, filename)`: Protokolliert die Testergebnisse in einer CSV-Datei.
- `rgba_loader(path)`: Lädt ein Bild im RGBA-Format.
- `train_model(data_dir, device, epochs, modelname)`: Trainiert das Modell mit den angegebenen Parametern.
- `test_model(test_data_dir, device, modelname, logfile)`: Testet das Modell auf einem Testdatensatz und speichert die Ergebnisse.
- `get_device(preferred_device)`: Wählt das bevorzugte Gerät (CPU, CUDA, MPS) für die Berechnungen aus.
- `copy_misclassified_images(csv_file, source_dir, target_dir)`: Kopiert falsch klassifizierte Bilder in ein Zielverzeichnis.
- `cleanup(path)`: Löscht den angegebenen Pfad.

Diese Funktionen und Klassen ermöglichen das Training, die Validierung und die Inferenz eines CNN-Modells zur Geschlechtsklassifikation von Gesichtsaufnahmen.


## Lösungsansätze im Vergleich
### Training und Evaluation


## Fazit