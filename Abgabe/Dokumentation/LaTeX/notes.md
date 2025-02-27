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
Der Datensatz für dieses Projekt besteht aus den Bildern und den dazugehörigen Masken, sowie der tagging.json Datei. Auf dieser JSON-Datei sind über die Dateinamen jeweils das Geschlecht der abgebildeten Person zugeordnet. Hiermit lässt  sich also der Datensatz in die Kategorien "männlich" und "weiblich" einteilen.
### Herausforderungen
Das Verarbeiten der Bilder auf einen gemeinsamen Nenner bietet eine gewisse Herausforderung. Hier ist jeweils aus der Maske die Information für die Position der Augen und des Mundes zu bestimmen. Mithilfe dieser Informationen werden die Bilder auf eine gemeinsame Bildgröße Verkleinert und die Augen und der Mund auf die selbe Position gebracht. Dadurch wird gewährleistet, dass das neuronale Netzwerk diese drei Punkte als Anhaltspunkte für den Vergleich und das Einlernen der Unterscheidungsmerkmale nutzen kann. 


## Code
Hier geht's um den eigentlichen Code (vielleicht mit Code-Snippets)
### Segmentierung
Der Code in `segmentierung.py` führt verschiedene Bildverarbeitungsaufgaben durch, um die person im Vordergrund zu isolieren und auf eine gemeinsame Zielmatrix zu transformieren. Die Hauptfunktionen umfassen:

- `freistellen(image, segm)`: Segmentiert das Bild basierend auf einer Maske und erstellt ein RGBA-Bild.
- `transformation(image, segm, eye_dist=70)`: Transformiert das Bild, um Augen und Mund auf eine feste Position zu bringen.
- `get_corners(segm)`: Bestimmt die Positionen der Augen und des Mundes aus der Maske.
- `get_single_center(segm, value)`: Findet das Zentrum eines bestimmten Segments im Bild.
- `get_lastpoint(centr)`: Berechnet den letzten Punkt basierend auf den Positionen der Augen.

Diese Funktionen helfen dabei, die Bilder auf eine einheitliche Größe zu bringen und die relevanten Gesichtspunkte zu normalisieren, um sie für ein neuronales Netzwerk vorzubereiten.

### Preprocessing
### CNN Modell


## Lösungsansätze im Vergleich
### Training und Evaluation


## Fazit