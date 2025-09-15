from utils.parser import load_config
from modules.postprocess import *
from modules.MaxEnt_classification import *
from modules.data_processing import *
from geo_utils.raster_utils import *
from geo_utils.vector_utils import *
import os

def main():
    config = load_config()

    # Output folder definieren
    output_folder_path = config["output_folder"]
    temp_folder = "temp_folder"
    temp_folder_path = os.path.join(output_folder_path, temp_folder)
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)

    # Parameter als Variablen definieren
    result_last_year = config["result_last_year_summer"]

    # MaxEnt Klassifikation
    classification_path = os.path.join(temp_folder_path, "classification.tif")
    classification_coreg_path = os.path.join(temp_folder_path, "classification_coreg_path.tif")
    if config["maxent"]["classification"] is None:
        run_maxent(config["force"]["ndvi"], config["force"]["ndwi"], config["force"]["nbr"], config["force"]["dswi"],
                   config["force"]["swir1"], config["maxent"]["training_points"], config["maxent"]["class_attribute"],
                   classification_path)
        if config["calc_disturbence"]:
            classification_coreg = co_registration(result_last_year, classification_path, "nearest",
                                                   classification_coreg_path)
    else:
        classification_coreg = co_registration(result_last_year, config["maxent"]["classification"], "nearest",
                                               classification_coreg_path)

    if config["calc_disturbence"]:
        # Co-registration der benötigten Raster mit dem Ergebnis des vergangenen Jahres
        harmonic_coreg_path = os.path.join(temp_folder_path, "harmonic_coreg_path.tif")
        analyseflaeche_coreg_path = os.path.join(temp_folder_path, "analyseflaeche_coreg_path.tif")

        harmonic_coreg = co_registration(result_last_year, config["harmonic_result"], "nearest", harmonic_coreg_path)
        analyseflaeche_coreg = co_registration(result_last_year, config["analyseflaeche"], "nearest", analyseflaeche_coreg_path)

        # Berechnung Schadflächen
        disturbence_current_year = calculate_disturbance(harmonic_coreg, analyseflaeche_coreg, classification_coreg, config["modus"], output_folder_path)

        # Berechnung der Differenz
        if config["bundesland"] == "thueringen":
            disturbence_change = calculate_disturbance_change(config["result_last_year_summer"], config["result_last_year_spring"], disturbence_current_year, analyseflaeche_coreg, config["modus"], output_folder_path)

            # Vektorisieren der Differenzberechnung
            if config["vectorizing"]:
                # Vektorisieren und filtern des disturbence change Rasters
                vectorize_raster(disturbence_change, config["vectorize"]["min_area"])

if __name__ == "__main__":
    main()
