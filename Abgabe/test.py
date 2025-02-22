import cv2
import numpy as np

def segment_filter(image_path): #, output_path):
    # Bild laden
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Maske erstellen für die gewünschten Werte
    mask = (image == 5) | (image == 6) | (image == 7)
    
    # Neues Bild initialisieren mit Schwarz
    result = np.zeros_like(image, dtype=np.uint8)
    
    # Gewünschte Bereiche auf Weiß setzen
    result[mask] = 255
    
    # Konturen finden
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Schwerpunkte der drei Segmente bestimmen
    moments = [cv2.moments(cnt) for cnt in contours]
    centroids = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) for m in moments if m['m00'] != 0]
    
    if len(centroids) != 3:
        raise ValueError("Nicht genau drei Segmentschwerpunkte gefunden!")
    
    # Zielpunkte für das gleichschenklige Dreieck berechnen
    base_mid = ((centroids[0][0] + centroids[1][0]) // 2, (centroids[0][1] + centroids[1][1]) // 2)
    height = int(np.linalg.norm(np.array(centroids[2]) - np.array(base_mid)))
    top_point = (base_mid[0], base_mid[1] - height)
    
    target_pts = np.float32([centroids[0], centroids[1], top_point])
    src_pts = np.float32(centroids)
    
    # Transformation berechnen
    matrix = cv2.getAffineTransform(src_pts, target_pts)
    warped = cv2.warpAffine(result, matrix, (image.shape[1], image.shape[0]))
    
    # Bounding Box berechnen und Bild zuschneiden
    x_min = min(p[0] for p in target_pts) - 20
    x_max = max(p[0] for p in target_pts) + 20
    y_min = min(p[1] for p in target_pts) - 20
    y_max = max(p[1] for p in target_pts) + 20
    
    cropped = warped[max(0, y_min):min(image.shape[0], y_max), max(0, x_min):min(image.shape[1], x_max)]
    
    # Bild speichern
    # cv2.imwrite(output_path, cropped)
    return cropped
# Beispielaufruf
# segment_filter('input.png', 'output.png')
