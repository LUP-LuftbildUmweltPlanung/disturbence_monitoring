import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from datetime import datetime
from pathlib import Path

def calculate_disturbance(harmonic_model_path, analyseflaeche_path, classification_path, output_folder_path):
    print("Berechne Schadflächen für das aktuelle Jahr..")
    # --- 2. Raster öffnen und prüfen ---
    with rasterio.open(harmonic_model_path) as src_model, \
         rasterio.open(analyseflaeche_path) as src_ana, \
         rasterio.open(classification_path) as src_klass:

        # CRS prüfen
        #if not (src_model.crs == src_ana.crs == src_klass.crs):
        #    raise ValueError("Die Raster haben unterschiedliche Koordinatensysteme!")

        # Shape prüfen
        #if not (src_model.shape == src_ana.shape == src_klass.shape):
        #    raise ValueError("Die Raster haben unterschiedliche Dimensionen!")

        # Daten laden
        model = src_model.read(1)
        ana = src_ana.read(1)
        klass = src_klass.read(1)

        # Leeres Ausgaberaster
        result = np.full(model.shape, 0, dtype=np.int32)

        # Regel 1: Stehend abgestorben
        mask1 = (model < -50) & (ana == 1) & (klass == 3)
        result[mask1] = 3

        # Regel 2: Freifläche
        mask2 = (model < -50) & (ana == 1) & (klass == 2)
        result[mask2] = 2

        # Regel 3: Immergrüner Nadelwald vital
        mask3 = (ana == 1) & ~mask1 & ~mask2
        result[mask3] = 1

        # Regel 4: Sonstiger Wald
        mask4 = ((ana == 2) | (ana == 3)) & ~mask1 & ~mask2
        result[mask4] = 4

        # Optional: alle anderen Masken bleiben NaN (z.B. außerhalb Analysefläche)

        # --- 3. Ergebnis speichern ---
        out_meta = src_model.meta.copy()
        out_meta.update({
            "dtype": "int32",
            "nodata": 0
        })

        # create filename for result and save raster
        year = str(datetime.now().year)
        output_folder = Path(output_folder_path)
        output_path = output_folder / f"disturbance_monitoring_{year}.tif"
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print("Klassifizierungs-Raster erfolgreich gespeichert:", output_path)
    return output_path

def calculate_disturbance_change(result_last_year_path, result_current_year_path, output_folder_path):
    print("Berechne Veränderung der Schadflächen..")
    # --- Raster öffnen ---
    with rasterio.open(result_last_year_path) as src_24, rasterio.open(result_current_year_path) as src_25:
        # CRS- & Shape-Prüfung & ggf. anpassen
        r24 = src_24.read(1)
        r25 = src_25.read(1)

        # Leeres Ergebnisraster (Standard: NaN)
        result = np.full(r24.shape, np.nan, dtype=np.float32)

        # Regel 1: Klasse 1
        mask1 = (r24 == 1) & (r25 == 1)
        result[mask1] = 1

        # Regel 2: Klasse 4
        mask4 = (r24 == 2) | (r24 == 3) | (r24 == 4)
        result[mask4] = 4

        # Regel 3: Klasse 2
        mask2 = (r24 == 1) & (r25 == 2)
        result[mask2] = 2

        # Regel 4: Klasse 3
        mask3 = (r24 == 1) & (r25 == 3)
        result[mask3] = 3

        # Regel 5: Klasse x
        mask5 = (r24 == 1) & (r25 == 4)
        result[mask5] = 4

        # --- Ergebnis speichern ---
        out_meta = src_24.meta.copy()
        out_meta.update({
            "dtype": "float32",
            "nodata": np.nan
        })

        # create filename for result and save raster
        year = str(datetime.now().year)
        output_folder = Path(output_folder_path)
        output_path = output_folder / f"disturbance_change_{year}.tif"
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print(f"Ergebnis gespeichert unter: {output_path}")
    return output_path
