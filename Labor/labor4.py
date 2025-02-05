import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import math

def random_colorgen():
    """
    Generiert eine zufällige Farbe im BGR-Format.
    """
    b = np.round(np.random.default_rng().random()*255)
    g = np.round(np.random.default_rng().random()*255)
    r = np.round(np.random.default_rng().random()*255)
    return (int(b), int(g), int(r))

def get_radius_and_angle(pix, cx, cy, bins):
    """
    Berechnet den Radius und den Winkel eines Punktes relativ zum Schwerpunkt (cx, cy).
    
    :param pix: Pixel-Koordinaten
    :param cx: Mittelpunkt X-Koordinate
    :param cy: Mittelpunkt Y-Koordinate
    :param bins: Anzahl der Winkelunterteilungen
    :return: Radius und Winkelindex
    """
    relx = pix[0][0] - cx
    rely = pix[0][1] - cy
    
    radius = int(np.sqrt(np.square(relx) + np.square(rely)))
    a = round((np.arctan2(relx, rely) + np.pi) / (2 * np.pi) * bins)
    
    return radius, a

def konturanalyse(image):
    """
    Analysiert die Konturen eines Bildes, teilt es in Segmente ein und bestimmt die Spitzenanzahl.
    """
    cf = json.load(open("ImageProcessingGUI.json", 'r'))  # Lade Konfigurationswerte aus JSON
    
    h, w = image.shape[0:2]
    
    # Neues leeres Bild erstellen
    newimage = np.uint8(np.zeros((h, w, 3)))
    
    # Umwandlung in Graustufen und Binärbild
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, cf["val1"], cf["val2"], cv2.THRESH_BINARY)
    
    # Konturen finden
    contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    bins = 90  # Winkelauflösung (360° / bins = 4° Schritte)
    img_cluster = int(np.ceil(np.sqrt(len(contours))))  # Rastergröße berechnen
    
    # Größe der Rasterrechtecke bestimmen
    rect_width = w // img_cluster
    rect_height = h // img_cluster
    
    # Raster-Offsets erstellen
    offsets = [(i, j) for i in range(img_cluster) for j in range(img_cluster)]
    
    for idx, cont in enumerate(contours):
        # Berechnung des Massenschwerpunkts der Kontur
        M = cv2.moments(cont)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Bestimme Position im Raster
        offset_x, offset_y = offsets[idx]
        start_x = offset_x * rect_width
        start_y = offset_y * rect_height
        
        # Teilbereich des Bildes auswählen
        cropped_image = newimage[start_y:start_y + rect_height, start_x:start_x + rect_width]
        
        # Zufällige Farbe für die Kontur bestimmen
        color = random_colorgen()
        
        # Marker am Schwerpunkt zeichnen
        cv2.drawMarker(image_binary, (cx, cy), (255, 255, 255))
        
        # Bestimme, wie viele Punkte in jeder Gruppe analysiert werden (Berechnung so, dass die Anzahl 90 nicht überschritten wird)
        chunk_size = math.ceil(len(cont) / 90)
        
        # Matrix zur Speicherung von Radius und Winkel
        matrix = []
        
        for j, pix in enumerate(cont):
            radius, a = get_radius_and_angle(pix, cx, cy, bins)
            matrix.append([radius, a])
        
        # Reduziere die Anzahl der Messpunkte durch Mittelwertbildung
        new_matrix = []
        for i in range(0, len(matrix), chunk_size):
            chunk = matrix[i:i + chunk_size]
            avg_radius = sum(row[0] for row in chunk) / len(chunk)
            avg_angle = chunk[0][1]
            new_matrix.append([avg_radius, avg_angle])
        
        # Zeichne Histogramm der Kontur
        for j, (avg_radius, avg_angle) in enumerate(new_matrix):
            point1 = (int(avg_angle * cropped_image.shape[1] / bins), cropped_image.shape[0] - 1)
            point2 = (int((avg_angle + chunk_size) * cropped_image.shape[1] / bins), cropped_image.shape[0] - int(2 * avg_radius))
            cv2.rectangle(cropped_image, point1, point2, color, thickness=cv2.FILLED)
        
        # Spitzen zählen (lokale Maxima)
        peak_count = 0
        for i in range(1, len(new_matrix) - 1):
            if new_matrix[i][0] > new_matrix[i - 1][0] and new_matrix[i][0] > new_matrix[i + 1][0]:
                peak_count += 1
                point1 = (int(new_matrix[i][1] * cropped_image.shape[1] / bins), cropped_image.shape[0] - 1)
                point2 = (int((new_matrix[i][1] + chunk_size) * cropped_image.shape[1] / bins), cropped_image.shape[0] - int(2 * new_matrix[i][0]))
                cv2.rectangle(cropped_image, point1, point2, (0, 0, 255), thickness=cv2.FILLED)
            
        print(f"Spitzenanzahl: {peak_count}")
    
        # Extrahiere X- und Y-Werte
        x_values = np.array([row[1] for row in new_matrix])
        y_values = np.array([row[0] for row in new_matrix])  # Annahme: Die erste Spalte enthält die relevanten Werte

        # Polynomielles Fitten (z. B. 4. Grad)
        degree = 30
        coeffs = np.polyfit(x_values, y_values, degree)
        poly_func = np.poly1d(coeffs)

        # Berechnung der Werte der Funktion
        x_fine = np.linspace(x_values.min(), x_values.max(), 500)
        y_fine = poly_func(x_fine)

        # Erste und zweite Ableitung
        dy_dx = np.polyder(poly_func)  # Erste Ableitung
        d2y_dx2 = np.polyder(dy_dx)  # Zweite Ableitung

        # Nullstellen der ersten Ableitung (potenzielle Extrema)
        roots = np.roots(dy_dx)

        # Filtere nur reale Werte im gültigen Bereich
        maxima_x = []
        maxima_y = []
        for r in roots:
            if np.isreal(r) and x_values.min() <= r.real <= x_values.max():
                r_real = r.real
                if np.polyval(d2y_dx2, r_real) < 0:  # Zweite Ableitung prüfen
                    maxima_x.append(r_real)
                    maxima_y.append(poly_func(r_real))

        print(maxima_x)
        # Plot zur Visualisierung
        # plt.plot(x_values, y_values, 'bo', label="Originalwerte")
        plt.plot(x_fine, y_fine, 'r-', label="Polynomial Fit")
        plt.scatter(maxima_x, maxima_y, color='green', marker='o', label="Lokale Maxima")
        plt.legend()
        plt.show()
    return image_binary, newimage

def run(image, result, settings=None):
    """
    Führt die Konturanalyse durch und speichert die Ergebnisse in der result-Liste.
    """
    data = np.sin(np.arange(0, 100, 0.5))
    data2, data1 = konturanalyse(image)
    result.append({"name": "Plot", "data": data1})
    result.append({"name": "Marker", "data": data2})

if __name__ == '__main__':
    matplotlib.use('Agg')  # Verhindert GUI-Fehler in nicht-interaktiven Umgebungen
    image = cv2.imread("Images\Objekte.png")
    result = []
    run(image, result)
    
    for ele in result:
        if len(ele["data"].shape) == 1:
            fig = plt.figure(num=ele["name"])
            plt.plot(ele["data"])
            fig.tight_layout()
            plt.grid(True)
            plt.xlim(0.0, len(ele["data"]) - 1)
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer._renderer)
            cv2.imshow(ele["name"], data)
        else:
            cv2.imshow(ele["name"], ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()