import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.interpolate import griddata  # WICHTIG: Dieser Import ist neu!
import scipy.ndimage as ndimage

def convert_piv_txt_to_bin(txt_folder_str, output_folder_str):
    folder = Path(txt_folder_str)
    output_folder = Path(output_folder_str)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    prefix = "Serie_"
    files = sorted(folder.glob(prefix + "*.txt"))
    
    if not files:
        print(f"Keine Textdateien in {folder} gefunden!")
        return

    print(f"{len(files)} Dateien gefunden. Starte Vorab-Konvertierung...")
    
    # Temporäre Liste für die extrahierten Datenpunkte
    all_features = []

    for i, file in enumerate(files):
        # 1. Datei laden (mit Semikolon!)
        df = pd.read_csv(file, sep=';', comment='#', header=None)
        
        x_coords = df.iloc[:, 0].values
        y_coords = df.iloc[:, 1].values
        u_vel = df.iloc[:, 2].values  # Vx
        v_vel = df.iloc[:, 3].values  # Vy

        # 2. Auf 64x64 Raster interpolieren (Cropping + Subsampling Schritt 1)
        grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):64j, 
                                  min(y_coords):max(y_coords):64j]

        u_grid = griddata((x_coords, y_coords), u_vel, (grid_x, grid_y), method='cubic')
        v_grid = griddata((x_coords, y_coords), v_vel, (grid_x, grid_y), method='cubic')
        
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        v_grid = np.nan_to_num(v_grid, nan=0.0)

        # 3. Vorticity (Wirbelstärke) berechnen
        # gradient() gibt [Ableitung_nach_Y, Ableitung_nach_X] zurück
        du_dy, du_dx = np.gradient(u_grid)
        dv_dy, dv_dx = np.gradient(v_grid)
        vorticity_64 = dv_dx - du_dy

        # 4. Smoothing (Glätten) wie im Paper
        # Sigma=1.0 ist ein guter Standardwert für leichten Gaussian Blur
        vorticity_smoothed = ndimage.gaussian_filter(vorticity_64, sigma=1.0)

        # 5. Zweites Subsampling auf 32x32 (wir nehmen einfach jeden zweiten Pixel)
        vorticity_32 = vorticity_smoothed[::2, ::2]

        # 6. Flachklopfen -> EXAKT 1024 Dimensionen
        data_1024 = vorticity_32.reshape(-1)
        
        all_features.append(data_1024)
        
        if (i + 1) % 10 == 0:
            print(f"Datei {i+1}/{len(files)} verarbeitet...")

    # Zu einer großen Matrix zusammenfügen
    npdata = np.array(all_features) # Form: [Anzahl_Dateien, 1024]
    
    print(f"\nExtraktion abgeschlossen. Matrix-Form: {npdata.shape}")
    
    # Zentrieren (Mittelwert abziehen)
    npdata = npdata - npdata.mean(axis=0)
    
    # Standardabweichung für die spätere Normalisierung berechnen
    std = npdata.std(axis=0)
    
    # Train/Test Split (1/3 für Testdaten reservieren)
    n_test = max(1, npdata.shape[0] // 3)
    train_data = npdata[0:-n_test]
    test_data = npdata[-n_test:]

    # Speichern
    np.save(output_folder / "piv_train.npy", train_data.astype(np.float32))
    np.save(output_folder / "piv_test.npy", test_data.astype(np.float32))
    np.save(output_folder / "piv_std.npy", std.astype(np.float32))
    
    print("✅ Fertig! Daten wurden erfolgreich komprimiert und gespeichert in:", output_folder)

if __name__ == "__main__":
    # Passe diese Pfade an deinen Rechner an
    convert_piv_txt_to_bin(
        txt_folder_str="data/newPIV/dataverse_files", 
        output_folder_str="data/preprocessed_piv"
    )