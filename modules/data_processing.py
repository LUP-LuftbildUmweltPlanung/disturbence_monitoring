import rasterio
import geopandas as gpd
from rasterio import features
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
import numpy as np
import os


def vectorize_raster(raster_path, min_area=None, output_path=None):
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        nodata_val = src.nodata

    # Vectorize raster
    shapes = features.shapes(image, mask=None, transform=transform)

    # Create value list of geometries
    geometries = []
    raster_values = []

    for geom, value in shapes:
        if np.isnan(value):
            pass
        elif value != nodata_val:  # filtering nodata values
            geometries.append(shape(geom))
            raster_values.append(value)

    # Create geodata frame and add raster values
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf.set_crs(src.crs, allow_override=True, inplace=True)
    gdf['raster_val'] = raster_values

    if min_area is not None:
        # Schritt 4: Flächen berechnen und filtern
        gdf['area'] = gdf.geometry.area
        # Schritt 5: Entfernen von kleinen Flächen
        min_area = min_area + 1
        gdf = gdf[gdf['area'] >= min_area]

    # Optional: Geodataframe speichern
    if output_path is None:
        basename, _ = os.path.splitext(raster_path)
        output_path = basename + "_vectorized.shp"

    gdf.to_file(output_path)
