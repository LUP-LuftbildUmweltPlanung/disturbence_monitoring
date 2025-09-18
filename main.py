from utils.parser import load_config
from utils.helper import *
from modules.postprocess import *
from modules.maxent_classification import *
from modules.data_processing import *
from geo_utils.raster_utils import *
import os
from datetime import datetime

def main():
    config = load_config()

    # Output folder definieren
    output_folder_path = config["output_folder"]
    temp_folder = "temp_folder"
    temp_folder_path = os.path.join(output_folder_path, temp_folder)
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    year = str(datetime.now().year)

    # Parameter und Variablen definieren
    result_last_year = config["result_last_year_summer"]
    modus = config["modus"]
    classification_path = os.path.join(temp_folder_path, "classification.tif")
    ndvi_threshold = config["ndvi_threshold"]
    classification_filtered_path = os.path.join(temp_folder_path, f"classification_filtered_{ndvi_threshold}.tif")
    classification_coreg_path = os.path.join(temp_folder_path, "classification_coreg.tif")
    harmonic_coreg_path = os.path.join(temp_folder_path, "harmonic_coreg.tif")
    analyseflaeche_coreg_path = os.path.join(temp_folder_path, "analyseflaeche_coreg.tif")
    disturbence_path = os.path.join(output_folder_path, f"disturbance_monitoring_{year}_{modus}.tif")
    difference_path = os.path.join(output_folder_path, f"disturbance_change_{year}_{modus}.tif")

    # MaxEnt Klassifikation
    if config["maxent"]["classification"] is None:
        run_maxent(config["force"]["ndvi"], config["force"]["ndwi"], config["force"]["nbr"], config["force"]["dswi"],
                   config["force"]["swir1"], config["maxent"]["training_points"], config["maxent"]["class_attribute"],
                   classification_path)
        if config["calc_disturbence"]:
            co_registration(result_last_year, classification_path, "nearest",
                                                   classification_coreg_path)
        hold_point(config, "Klassifikation berechnet. Ergebnisse prüfen. Weiter mit Enter, Abbruch mit 'n'")

    if config["postprocess_classification"]:
        # Co-registration der Analysefläche
        analyseflaeche = co_registration(config["maxent"]["classification"], config["analyseflaeche"], "nearest")

        # Filtern der Klassifikation mit NDVI Schwellenwert
        filter_classification(config["force"]["ndvi"], config["maxent"]["classification"], analyseflaeche, ndvi_threshold, classification_filtered_path)

        hold_point(config, "Klassifikation gefiltert. Ergebnisse prüfen. Wenn das Ergebnis für die Schadflächenberechnung genutzt werden soll "
                           "bitte Abbruch mit 'n' und anschließend den Pfad zur Klassifikation in den Parametern aktualisieren.")

    if config["calc_disturbence"]:
        # Co-registration der benötigten Raster mit dem Ergebnis des vergangenen Jahres
        co_registration(result_last_year, config["harmonic_result"], "nearest", harmonic_coreg_path)
        co_registration(result_last_year, config["analyseflaeche"], "nearest", analyseflaeche_coreg_path)
        co_registration(result_last_year, config["maxent"]["classification"], "nearest", classification_coreg_path)

        # Berechnung Schadflächen
        calculate_disturbance(harmonic_coreg_path, analyseflaeche_coreg_path, classification_coreg_path, config["modus"], disturbence_path)

        hold_point(config, "Schadflächen berechnet. Ergebnisse prüfen. Weiter mit Enter, Abbruch mit 'n'")

    # Berechnung der Differenz
    if config["calc_difference"]:
        calculate_disturbance_change(config["result_last_year_summer"], config["result_current_year_spring"], disturbence_path, config["modus"], difference_path)

        hold_point(config, "Differenz berechnet. Ergebnisse prüfen. Weiter mit Enter, Abbruch mit 'n'")

    # Vektorisieren der Differenzberechnung
    if config["vectorize"]:
        # Vektorisieren und filtern des disturbence change Rasters
        vectorize_raster(difference_path, config["min_area"])

if __name__ == "__main__":
    main()
