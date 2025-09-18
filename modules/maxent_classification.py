import geopandas as gpd
import numpy as np
from tqdm import tqdm
import os
import rasterio
from rasterio import features
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import Counter


def run_maxent(ndvi_path, ndwi_path, nbr_path, dswi_path, sw1_path, training_points, class_attribute, output_path=None):
    raster_paths = [nbr_path, ndvi_path, ndwi_path, dswi_path, sw1_path]
    n_splits = 5
    # ---------- 1. Daten vorbereiten ----------
    # Raster öffnen
    rasters = [rasterio.open(r) for r in raster_paths]
    assert all((r.shape == rasters[0].shape for r in rasters)), "Alle Raster müssen gleiche Form haben"

    # CRS abgleichen
    raster_crs = rasters[0].crs
    gdf = gpd.read_file(training_points).to_crs(raster_crs)
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # Rasterwerte extrahieren
    samples = [np.array([val[0] for val in r.sample(coords)]) for r in rasters]
    X = np.stack(samples, axis=1)
    y = gdf[class_attribute].astype(int).values

    # ---------- 2. Kreuzvalidierung vorbereiten ----------
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        # Splitten und Skalieren
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modell trainieren
        clf = LogisticRegression(
            penalty='l2',
            solver='saga',
            multi_class='multinomial',
            #class_weight='balanced',
            max_iter=2000,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = clf.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, digits=3, output_dict=True)

        all_reports.append(report)

    # ---------- 3. Durchschnittliche Bewertung (optional) ----------
    # z. B. mittlere F1-Score über alle Folds berechnen
    classes = sorted(np.unique(y))
    avg_f1 = {cls: np.mean([rep[str(cls)]['f1-score'] for rep in all_reports]) for cls in classes}

    print("\n📊 Mittlere F1-Scores pro Klasse:")
    for cls, score in avg_f1.items():
        print(f"  Klasse {cls}: {score:.3f}")

    # ---------- 3. Finales Modell auf allen Daten ----------
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)

    # ---------- 4. Rasterklassifikation ----------
    meta = rasters[0].meta.copy()
    meta.update(dtype='uint8', count=1, nodata=0)
    nodata_values = [r.nodata for r in rasters]

    height, width = rasters[0].shape
    output = np.zeros((height, width), dtype=np.uint8)
    raster_data = [r.read(1) for r in rasters]

    for row in tqdm(range(height), desc="Klassifiziere Rasterzeilen"):
        row_data = [band[row, :] for band in raster_data]
        row_stack = np.stack(row_data, axis=1)

        mask = np.zeros(row_stack.shape[0], dtype=bool)
        for i, nodata in enumerate(nodata_values):
            band_row = row_stack[:, i]
            mask |= (band_row == nodata) | np.isnan(band_row)

        valid_data = row_stack[~mask]

        if valid_data.shape[0] > 0:
            valid_scaled = scaler.transform(valid_data)
            pred = clf.predict(valid_scaled)
            output[row, ~mask] = pred
        output[row, mask] = 0  # NoData

    # ---------- 5. GeoTIFF speichern ----------
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(output, 1)

    print(f"\n✅ Klassifikation abgeschlossen. Ergebnis gespeichert unter: {output_path}")
    return output_path
