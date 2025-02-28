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
### Entwicklungswerkzeuge: 
Python ist eine leistungsfähige, interpretierte Programmiersprache, die sich sehr gut für die Bildverarbeitung eignet. Die Bibliothek OpenCV bietet hier umfangreiche Unterstützung für Bildanalyse, Filterung, Merkmalsextraktion und Segmentierung. Die Kombination mit Deep-Learning-Frameworks wie z.B. PyTorch ermöglicht die Entwicklung fortschrittlicher Bildverarbeitungsalgorithmen, darunter Objekterkennung, Segmentierung und Bildklassifikation. Aufgrund seiner einfachen Syntax, plattformübergreifenden Kompatibilität und starken Community ist Python eine bevorzugte Wahl für Forschung und industrielle Anwendungen im Bereich der computergestützten Bildverarbeitung.

### Bildverarbeitung:

### Klassifikation:



### Möglichkeiten:


## Datensatz
Hier gehts um die Bilder lol.
### Allgemein
Der Datensatz für dieses Projekt besteht aus den Bildern und den dazugehörigen Masken, sowie der tag.json Datei. Auf dieser JSON-Datei sind über die Dateinamen jeweils das Geschlecht der abgebildeten Person zugeordnet. Hiermit lässt  sich also der Datensatz in die Kategorien "männlich" und "weiblich" einteilen.

Der Datensatz besteht aus ca. 2000 Bildern unterschiedlicher Qualität und Perspektive, die als JPEG-Dateien vorliegen. Zu jedem Bild existiert eine zugehörige Segmentierungsmaske im PNG-Format, welche die relevanten Bildbereiche kennzeichnet. Zusätzlich enthält der Datensatz eine tags.json-Datei, die für jedes Bild das Geschlecht der abgebildeten Person angibt. Die Geschlechterverteilung innerhalb des Datensatzes umfasst ca. 1200 Bilder von Männern und 800 Bilder von Frauen. Dieser Datensatz eignet sich insbesondere für Anwendungen im Bereich der Bildsegmentierung, geschlechtsspezifischen Bildanalysen und Deep-Learning-gestützten Erkennungsaufgaben.

### Herausforderungen
Das Verarbeiten der Bilder auf einen gemeinsamen Nenner bietet eine gewisse Herausforderung. Hier ist jeweils aus der Maske die Information für die Position der Augen und des Mundes zu bestimmen. Mithilfe dieser Informationen werden die Bilder auf eine gemeinsame Bildgröße Verkleinert und die Augen und der Mund auf die selbe Position gebracht. Dadurch wird gewährleistet, dass das neuronale Netzwerk diese drei Punkte als Anhaltspunkte für den Vergleich und das Einlernen der Unterscheidungsmerkmale nutzen kann. 

Die Nutzung dieses Datensatzes bringt mehrere Herausforderungen mit sich. Die variierende Bildqualität und unterschiedlichen Perspektiven könnten die Konsistenz der Segmentierung beeinträchtigen und die Generalisierbarkeit von Modellen erschweren. Zudem besteht eine Ungleichverteilung der Geschlechter mit 1200 Bildern von Männern und 800 von Frauen, was zu Verzerrungen in geschlechtsspezifischen Analysen führen kann. Die Qualität und Konsistenz der Segmentierungsmasken ist ein weiterer kritischer Faktor, da ungenaue oder fehlerhafte Masken die Modellleistung negativ beeinflussen könnten. Auch die Labels in der tags.json-Datei könnten Ungenauigkeiten enthalten oder nicht-binäre Identitäten ausschließen, was die Anwendbarkeit in diversen Szenarien einschränkt. Darüber hinaus erfordert die Verarbeitung von 2000 Bildern und Masken erhebliche Rechenleistung und Speicherplatz, insbesondere bei hochauflösenden Daten. 

Falls die Bilder reale Personen zeigen, müssen zudem Datenschutzrichtlinien wie die DSGVO beachtet werden. 

Schließlich könnten je nach Anwendung weitere Herausforderungen auftreten, etwa wenn die Segmentierungsqualität oder Perspektivenvielfalt die Leistung eines Erkennungsmodells beeinträchtigt. Diese Aspekte sollten bei der Vorverarbeitung und Modellentwicklung sorgfältig berücksichtigt werden, um Verzerrungen zu minimieren und robuste Ergebnisse zu erzielen.


## Code
Hier geht's um den eigentlichen Code (ohne Code-Snippets)
### Segmentierung
Die Datei "preprocess.py" dient der Verarbeitung des Datensatzes. Hier werden gezielt Funktionen implementiert, um die gegebenen Ressourcen (Segmentierungsmaske) zu nutzen und die Personen auf den Bildern aufgrund dessen Freizustellen und in die gewünschte Position und Ausrichtung zu transformieren. 

Die Transformation erfolgt über die so genannte affine Transformation. Diese bewirkt, dass geometrische Merkmale nach der Transformation weiterhin erhalten bleiben. Somit bleiben beispielsweise parallele Linien weiterhin parallel.

Zur Berechnung der Transforamtionsmatrix werden die einzelnen Schwerpunkte der Augen und des Musndes benötigt. Diese dienen dem neuronalen Netzwerk später als Ankerpunkte, um für jedes Gesicht immer möglichst die gleiche Ausgangslage zu haben. Zudem wird mit angabe der "Shape" auch die Skalierung der Bilder auf eine einheitliche Größe gebracht.

(Neben der Transformation und dem Freistellen wird gemäß der Aufgabenstellung auch ein zusätzlicher Kanal eingefügt. Hier handelt es sich um den Alpha Kanal.)

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