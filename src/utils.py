from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio as rio
from shapely import Polygon
from rasterio.features import rasterize
from rasterio.io import DatasetReader


EARTH_RADIUS = 6_371_000  # m (Average Earth radius)
DEGREE_LENGTH = 2 * np.pi * EARTH_RADIUS / 360


def get_raster_profile(dem_path: Path) -> dict:
    src = rio.open(dem_path)
    return src.profile


def find_raster_resolution(src: DatasetReader) -> tuple[float,float] | None:
    assert src.crs, 'Missing or invalid CRS'

    x_res, y_res = src.res

    if src.crs.is_geographic:
        # image res in meters
        return (y_res * DEGREE_LENGTH,
                x_res * DEGREE_LENGTH * np.cos(
                np.deg2rad(src.meta['transform'][4])
                )
            )
    else: # Assuming src.crs.is_projected is True
        return y_res, x_res


def raster_extent(tif_path: Path) -> gpd.GeoDataFrame:
    '''Extracts [as GeoDataFrame] and saves [as .shp] raster extent'''
    img_scr = rio.open(tif_path)
    height, width = img_scr.shape
    row_list = [0, height, height, 0, 0]
    col_list = [0, 0, width, width, 0]
    # transform image corner coordinates into lon/lat
    lon_list, lat_list = rio.transform.xy(img_scr.meta['transform'], # type: ignore
                                          row_list, col_list)
    img_extent = Polygon(zip(lon_list, lat_list))
    # Poligonize image extent
    return gpd.GeoDataFrame(index=[0],
                            crs=img_scr.crs,
                            geometry=[img_extent]) # type: ignore


def sea_mask(dem_path, sea_mask_shp):
        extent = raster_extent(dem_path)
        shape = gpd.read_file(sea_mask_shp).dissolve()
        diff = extent.difference(shape)
        src = rio.open(dem_path)
        array_to_rast = rasterize(
                        shapes=diff.geometry,
                        out_shape=src.shape,
                        transform=src.meta['transform'],
                        all_touched=True)
        mask = array_to_rast == 0
        return mask
