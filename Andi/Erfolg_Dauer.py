import numpy as np
import matplotlib.pyplot as plt

# IEEE Standardfarben
ieee_blue = "#004393"
ieee_blue2 = "#138ADA"

# Beispiel-Daten
modelle = ["model35_3Layer_70px","model90_3Layer_70px", "model30_3Layer_100px", "model30_AUG_100px", "model40_AUG_4Layer_100px"]
testerfolg = [92.75, 90.47, 92.98, 91.22,99.49]  # In Prozent (0 bedeutet kein Wert verfügbar)
zeit_pro_epoche = [129, 386 , 107 ,117, 25]  # Zeit in Sekunden

# Positionen für die Balken
x = np.arange(len(modelle))  # X-Positionen für jedes Modell
breite = 0.35  # Breite der Balken
abstand = breite * 0.25  # Abstand zwischen den Balken (¼ der Breite)

# Erstellen des Diagramms
fig, ax1 = plt.subplots(figsize=(10, 6))

# Balken für Testerfolg (links)
balken1 = ax1.bar(x - (breite / 2 + abstand / 2), testerfolg, breite, label="Testerfolg (%)", color=ieee_blue)
ax1.set_ylabel("Testerfolg (%)", color=ieee_blue)
ax1.set_ylim(0, 110)  # Skala für Prozentwerte

# Zweite Achse für Zeit pro Epoche (rechts)
ax2 = ax1.twinx()
balken2 = ax2.bar(x + (breite / 2 + abstand / 2), zeit_pro_epoche, breite, label="Trainingsdauer (Min.)", color=ieee_blue2)
ax2.set_ylabel("Zrainingsdauer (Min.)", color=ieee_blue2)
ax2.set_ylim(0, max(zeit_pro_epoche) * 1.2)  # Skala leicht höher setzen

# Werte über die Balken schreiben
for i in range(len(modelle)):
    ax1.text(x[i] - (breite / 2 + abstand / 2), testerfolg[i] + 2, f"{testerfolg[i]}%", ha='center', fontsize=10, color=ieee_blue)
    ax2.text(x[i] + (breite / 2 + abstand / 2), zeit_pro_epoche[i] + 5, f"{zeit_pro_epoche[i]}min", ha='center', fontsize=10, color=ieee_blue2)

# Achsentitel & Labels
ax1.set_xticks(x)  # Zentriert die Labels unter den Gruppen
ax1.set_xticklabels(modelle, rotation=0, ha="center",fontsize=6)

# Titel noch weiter oben platzieren
plt.title("Vergleich: Testerfolg vs. Zeit pro Epoche", fontsize=14, fontweight='bold', pad=25)

# Legende mit mehr Abstand zum Titel setzen
ax1.legend(loc="upper left", bbox_to_anchor=(0.5, 1.075), ncol=2, frameon=False)
ax2.legend(loc="upper right", bbox_to_anchor=(0.5, 1.075), ncol=2, frameon=False)

# Diagramm anzeigen
plt.show()
