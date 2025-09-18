import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union

def filter_classification(ndvi_path, classification_path, analyseflaeche_path, ndvi_threshold, output_path):
    print("Filter Klassifikation mit NDVI threshold..")
    # --- Raster öffnen ---

    with rasterio.open(ndvi_path) as ndvi_raster, \
         rasterio.open(classification_path) as classification_raster, \
         rasterio.open(analyseflaeche_path) as ana_raster :

        ndvi = ndvi_raster.read(1)
        classification = classification_raster.read(1)
        ana = ana_raster.read(1)

        # Ergebnisraster initiieren
        result = classification.copy()
        nodata = 0

        # NoData überall durchreichen (vor dem Überschreiben markieren)
        mask_nodata = (ndvi == nodata) | (classification == nodata) | (ana == nodata)

        # Filtern der Klassifkation für Nadelwald
        result[(ndvi > ndvi_threshold) & (classification == 3) & (ana == 1) & ~mask_nodata] = 1

        # Filtern der Klassifkation für sonstige Waldfläche
        result[(ndvi > ndvi_threshold) & (classification == 3) & (ana > 1) & ~mask_nodata] = 4

        # optional: sicherstellen, dass dtype passt
        result = result.astype("uint8")

        # --- Ergebnis speichern ---
        out_meta = classification_raster.meta.copy()
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

def _shared_border_len(a, b):
    """Länge der gemeinsamen Grenze zweier Geometrien (0 wenn sie sich nicht berühren)."""
    inter = a.boundary.intersection(b.boundary)
    return inter.length if not inter.is_empty else 0.0


def vectorize_raster(
    raster_path,
    min_area=None,
    output_path=None,
    strict_int=True  # True => Fehler, wenn Werte nicht (nahezu) ganzzahlig sind
):
    # --- Raster lesen ---
    with rasterio.open(raster_path) as src:
        band1 = src.read(1)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs

    # --- Maske bauen (NoData ausblenden) ---
    if nodata is None:
        mask = None
    else:
        if isinstance(nodata, float) and np.isnan(nodata):
            mask = ~np.isnan(band1)
        else:
            mask = band1 != nodata

    # --- Raster -> Vektoren ---
    geoms = []
    vals = []
    for geom, val in features.shapes(band1, mask=mask, transform=transform):
        # Sicherheitsfilter: NoData raus (auch wenn mask schon gesetzt ist)
        if nodata is not None:
            if isinstance(nodata, float) and np.isnan(nodata):
                if isinstance(val, float) and np.isnan(val):
                    continue
            elif val == nodata:
                continue
        geoms.append(shape(geom))
        vals.append(val)

    # --- Werte als int absichern ---
    vals = np.asarray(vals)
    if np.issubdtype(vals.dtype, np.floating):
        # sind die Werte praktisch ganzzahlig? (z.B. 1.0, 2.0)
        if np.all(np.isclose(vals, np.round(vals), equal_nan=False)):
            vals = np.round(vals).astype(np.int64)
        else:
            if strict_int:
                raise ValueError(
                    "Rasterwerte sind nicht ganzzahlig. Setze strict_int=False, "
                    "wenn du trotzdem runden und als int speichern willst."
                )
            vals = np.round(vals).astype(np.int64)
    else:
        vals = vals.astype(np.int64)

    # --- GeoDataFrame ---
    gdf = gpd.GeoDataFrame({"class": vals}, geometry=geoms, crs=crs).reset_index(drop=True)

    # --- Kleine Flächen auf größere Nachbarn mergen ---
    if min_area and min_area > 0:
        gdf["area"] = gdf.geometry.area
        # Sicherheitslimit, um Endlosschleifen zu vermeiden
        max_iters = len(gdf) * 2
        it = 0

        while True:
            it += 1
            if it > max_iters:
                break

            small = gdf.index[gdf["area"] < min_area]
            if len(small) == 0:
                break

            i = int(small[0])
            geom_i = gdf.geometry.iloc[i]

            # Kandidaten per SIndex & bbox
            sidx = gdf.sindex
            cand = [j for j in sidx.intersection(geom_i.bounds) if j != i]

            # Echte Nachbarn (touch/intersect)
            touching = [j for j in cand
                        if geom_i.touches(gdf.geometry.iloc[j]) or geom_i.intersects(gdf.geometry.iloc[j])]

            # Ziel: längste gemeinsame Grenze -> sonst nächster Nachbar
            target = None
            if touching:
                best_len = -1.0
                for j in touching:
                    l = _shared_border_len(geom_i, gdf.geometry.iloc[j])
                    if l > best_len:
                        best_len = l
                        target = j
            else:
                # ggf. auf alle anderen ausweichen
                others = [j for j in range(len(gdf)) if j != i]
                if others:
                    dists = [(j, geom_i.distance(gdf.geometry.iloc[j])) for j in others]
                    target = min(dists, key=lambda x: x[1])[0]

            if target is None:
                # nichts Sinnvolles zu mergen
                break

            # Merge: Ziel behält Attribute
            new_geom = unary_union([gdf.geometry.iloc[target], geom_i])
            gdf.at[target, "geometry"] = new_geom
            gdf.at[target, "area"] = new_geom.area

            # kleines Polygon entfernen und Index neu aufsetzen
            gdf = gdf.drop(index=i).reset_index(drop=True)

        gdf = gdf.drop(columns="area")

    # --- Ausgabe ---
    if output_path is None:
        base, _ = os.path.splitext(raster_path)
        output_path = base + "_vectorized.shp"

    # GeoPackage oder Shapefile – wird automatisch anhand der Endung gewählt
    gdf.to_file(output_path)

    return output_path
