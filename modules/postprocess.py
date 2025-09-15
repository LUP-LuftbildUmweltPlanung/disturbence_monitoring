import numpy as np
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from datetime import datetime
from pathlib import Path
import rasterio

def calculate_disturbance(harmonic_model_path, analyseflaeche_path, classification_path, modus, output_folder_path):
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

        # create filename for result and save raster
        year = str(datetime.now().year)
        output_folder = Path(output_folder_path)
        output_path = output_folder / f"disturbance_monitoring_{year}_{modus}.tif"
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print("Klassifizierungs-Raster erfolgreich gespeichert:", output_path)
    return output_path

def calculate_disturbance_change(result_last_year_summer_path, result_current_year_spring, result_current_year_path, analyseflaeche_path, modus, output_folder_path):
    print("Berechne Veränderung der Schadflächen..")
    # --- Raster öffnen ---

    with rasterio.open(result_last_year_summer_path) as summer_raster, \
         rasterio.open(result_current_year_path) as current_raster:

        summer = summer_raster.read(1)
        current = current_raster.read(1)

        # Leeres Ergebnisraster (Standard: NaN)
        result = np.full(summer.shape, np.nan, dtype=np.uint8)

        if modus == "fruehjahr":
            # Regel 1: Klasse 1 Nadelwald vital
            mask1 = (summer == 1) & (current == 1)
            result[mask1] = 1

            # Regel 2: Klasse 4 sonstiger Wald (Freiflächen und abgestorbene Waldflächen werden zu sonstiger Wald)
            mask4 = (summer == 2) | (summer == 3) | (summer == 4)
            result[mask4] = 4

            # Regel 3: Klasse 2 neue Freiflächen auf Nadelwald
            mask2 = (summer == 1) & (current == 2)
            result[mask2] = 2

            # Regel 4: Klasse 3 neu stehend abgestorbener Nadelwald
            mask3 = (summer == 1) & (current == 3)
            result[mask3] = 3

            # Regel 5: Klasse 4 neuer sonstiger Wald (entstanden aus Nadelwald)
            mask5 = (summer == 1) & (current == 4)
            result[mask5] = 4

            # Regel 6: Klasse 1 neuer Nadelwald (entstanden aus sonstigen Wald))
            mask6 = (summer == 4) & (current == 1)
            result[mask6] = 1

        if modus == "sommer":
            with rasterio.open(result_current_year_spring) as spring_raster, \
                 rasterio.open(analyseflaeche_path) as src_ana:

                spring = spring_raster.read(1)
                ana = src_ana.read(1)

                ### Berechnung der Differenz für Nadelwald
                # Regel 1: Klasse 1 Nadelwald vital
                mask1 = (spring == 1) & (current == 1)
                result[mask1] = 1

                # Regel 2: Klasse 4 sonstiger Wald (Freiflächen und stehend abgestorben auf Nadelwald werden zu sonstiger Wald)
                mask4 = (spring == 2) | (spring == 3)
                result[mask4] = 4

                # Regel 3: Klasse 2 neue Freifläche auf Nadelwald
                mask2 = (spring == 1) & (current == 2)
                result[mask2] = 2

                # Regel 4: Klasse 3 neu stehend abgestorbener Nadelwald
                mask3 = (spring == 1) & (current == 3)
                result[mask3] = 3

                # Regel 5: Klasse 4 sonstiger Wald
                mask5 = (spring == 1) & (current == 4)
                result[mask5] = 4

                # Regel 6: Klasse 1 Nadelwald vital
                mask6 = (spring == 4) & (current == 1)
                result[mask6] = 1

                ### Berechnung der Differenz für Lärche und sonstige Waldflächen
                # Regel 2: Klasse 4 sonstiger Wald (Freiflächen und abgestorbene Waldflächen werden zu sonstiger Wald)
                mask7 = (summer == 2) | (summer == 3)
                result[mask7] = 4

                # Regel 3: Klasse 2 neue Freifläche auf sonstigen Wald
                mask8 = (summer == 4) & (current == 2)
                result[mask8] = 2

                # Regel 4: Klasse 3 neu stehend abgestorben auf sonstigen Wald
                mask9 = (summer == 4) & (current == 3)
                result[mask9] = 3

        # --- Ergebnis speichern ---
        out_meta = summer_raster.meta.copy()
        out_meta.update({
            "dtype": "uint8",
            "nodata": 255
        })

        # create filename for result and save raster
        year = str(datetime.now().year)
        output_folder = Path(output_folder_path)
        output_path = output_folder / f"disturbance_change_{year}.tif"
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(result, 1)

    print(f"Ergebnis gespeichert unter: {output_path}")
    return output_path
