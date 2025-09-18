import numpy as np
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from datetime import datetime
from pathlib import Path
import rasterio

def calculate_disturbance(harmonic_model_path, analyseflaeche_path, classification_path, modus, output_path):
    print("Berechne Schadflächen für das aktuelle Jahr..")
    # --- 2. Raster öffnen und prüfen ---
    with rasterio.open(harmonic_model_path) as src_model, \
         rasterio.open(analyseflaeche_path) as src_ana, \
         rasterio.open(classification_path) as src_klass:

        # Daten laden
        model = src_model.read(1)
        ana = src_ana.read(1)
        klass = src_klass.read(1)

        # Leeres Ausgaberaster
        result = np.full(ana.shape, 0, dtype=np.uint8)

        if modus == "fruehjahr":
            # Regel 1: Freiflächen auf Nadelwald
            mask1 = (model < -50) & (ana == 1) & (klass == 2)
            result[mask1] = 2

            # Regel 2: Stehend abgestorben auf Nadelwald
            mask2 = (model < -50) & (ana == 1) & (klass == 3)
            result[mask2] = 3

            # Regel 3: Immergrüner Nadelwald vital
            mask3 = (ana == 1) & ~mask1 & ~mask2
            result[mask3] = 1

            # Regel 4: Sonstiger Wald (Laubwald & Lärche)
            mask4 = ((ana == 2) | (ana == 3)) & ~mask1 & ~mask2
            result[mask4] = 4

        if modus == "sommer":
            # Regel 1: Freiflächen auf gesamter Analysefläche
            mask1 = (model < -50) & (ana > 0) & (klass == 2)
            result[mask1] = 2

            # Regel 2: Stehend abgestorben auf gesamter Analysefläche
            mask2 = (model < -50) & (ana > 0) & (klass == 3)
            result[mask2] = 3

            # Regel 3: Immergrüner Nadelwald vital
            mask3 = (ana == 1) & ~mask1 & ~mask2
            result[mask3] = 1

            # Regel 4: Sonstiger Wald (Laubwald & Lärche)
            mask4 = ((ana == 2) | (ana == 3)) & ~mask1 & ~mask2
            result[mask4] = 4

        # --- 3. Ergebnis speichern ---
        out_meta = src_model.meta.copy()
        out_meta.update({
            "dtype": "uint8",
            "nodata": 0
        })

        # save raster
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print("Klassifizierungs-Raster erfolgreich gespeichert:", output_path)
    return output_path

def calculate_disturbance_change(result_last_year_summer_path, result_current_year_spring, result_current_year_path, modus, output_path):
    print("Berechne Veränderung der Schadflächen..")
    # --- Raster öffnen ---

    with rasterio.open(result_last_year_summer_path) as summer_raster, \
         rasterio.open(result_current_year_path) as current_raster:

        summer = summer_raster.read(1)
        current = current_raster.read(1)

        # Ergebnisraster initiieren
        result = current.copy()
        nodata = 0

        if modus == "fruehjahr":

            # NoData überall durchreichen (vor dem Überschreiben markieren)
            mask_nodata = (current == nodata) | (summer == nodata)

            # im vergangenen Frühjahr erfasste Freiflächen und stehend abgestorben (nur auf Nadelwald)
            result[((summer == 2) | (summer == 3) | (summer == 4)) & ~mask_nodata] = 4

        if modus == "sommer":
            with rasterio.open(result_current_year_spring) as spring_raster:

                spring = spring_raster.read(1)

                # NoData überall durchreichen (vor dem Überschreiben markieren)
                mask_nodata = (current == nodata) | (spring == nodata) | (summer == nodata)

                # im vergangenen Frühjahr erfasste Freiflächen und stehend abgestorben (nur auf Nadelwald)
                result[((spring == 2) | (spring == 3)) & ~mask_nodata] = 4

                # im vergangenen Sommer erfasste Freiflächen und stehend abgestorben (auf Nadelwald & sonstigen Laubwald)
                result[((summer == 2) | (summer == 3)) & ~mask_nodata] = 4

        # optional: sicherstellen, dass dtype passt
        result = result.astype("uint8")

        # --- Ergebnis speichern ---
        out_meta = current_raster.meta.copy()
        out_meta.update({
            "dtype": "uint8",
            "nodata": 0,
            "count": 1,
            "compress": "lzw"
        })

        # save raster
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print(f"Ergebnis gespeichert unter: {output_path}")
    return output_path
