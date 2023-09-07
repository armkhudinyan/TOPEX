from typing import Optional, Union
from pathlib import Path
import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader


class Topex:

    def __init__(
            self,
            dem: np.ndarray,
            max_distance: float,
            interval: float,
            *,
            y_res: float,
            x_res: Optional[float] = None
    ) -> None:
        self.dem = dem
        self.max_distance = max_distance
        self.interval = interval
        self.y_res = y_res
        self.x_res = x_res if x_res else self.y_res

        self.pixels_per_interval_y = self.interval / self.y_res
        self.pixels_per_interval_x = self.interval / self.x_res
        self.pixels_per_interval_diag_y = self.interval * np.cos(np.pi / 4) / self.y_res
        self.pixels_per_interval_diag_x = self.interval * np.sin(np.pi / 4) / self.x_res

    def _distances(self, pixels_per_inteval: float) -> np.ndarray:
        '''List of distances to calculate the topex for each location '''
        return (np.arange(1, round(self.max_distance / self.interval) + 1
            ) * pixels_per_inteval).round().astype(int)

    def _topex_along_axis(self, dem: np.ndarray, axis: int=0) -> np.ndarray:
        # NOTE: Axis 2 here refers to any of the diagonal directions

        slope = np.full(dem.shape, np.pi / 2)
        height, width = dem.shape
        # NOTE: Distances per interval in number of rows and cols, not in meters
        distances: np.ndarray = (
                    self._distances(self.pixels_per_interval_y) if axis == 0
            else    self._distances(self.pixels_per_interval_x) if axis == 1
            else    np.array([(row,col) for row,col in zip(
                    self._distances(self.pixels_per_interval_diag_y),
                    self._distances(self.pixels_per_interval_diag_x))])
        )

        pad_size = (distances[-1] if axis < 2
            else   (distances[-1][0], distances[-1][1]))

        padded_dem: np.ndarray = (
                    np.pad(dem, ((pad_size,), (0,))) if axis == 0
            else    np.pad(dem, ((0,), (pad_size,))) if axis == 1
            else    np.pad(dem, ((pad_size[0],), (pad_size[1],)))
        )

        dist_meters = np.arange(self.interval, self.max_distance +
                                self.interval, self.interval)
        for i, distance in enumerate(distances): # Number of pixels
            if axis == 0:
                delta_height = dem - padded_dem[pad_size + distance:
                                                pad_size + distance + height, :]
            elif axis == 1:
                delta_height = dem - padded_dem[:, pad_size + distance :
                                                   pad_size + distance + width]
            else:
                num_rows, num_cols = distance
                pad_size_y, pad_size_x = pad_size

                delta_height = dem - padded_dem[
                    pad_size_y + num_rows : pad_size_y + num_rows + height,
                    pad_size_x + num_cols : pad_size_x + num_cols + width]

            angle = np.arctan(delta_height / dist_meters[i])
            slope = np.minimum(slope, angle)

        return slope * -1

    def north(self) -> np.ndarray:
        flip_dem = self.dem[::-1] # flip along 0 axis
        return self._topex_along_axis(flip_dem, axis=0)[::-1]

    def north_east(self) -> np.ndarray:
        flip_dem = self.dem[::-1] # flip along 0 axis
        return self._topex_along_axis(flip_dem, axis=2)[::-1]

    def east(self) -> np.ndarray:
        return self._topex_along_axis(self.dem, axis=1)

    def south_east(self) -> np.ndarray:
        return self._topex_along_axis(self.dem, axis=2)

    def south(self) -> np.ndarray:
        return self._topex_along_axis(self.dem, axis=0)

    def south_west(self) -> np.ndarray:
        flip_dem = self.dem[:, ::-1] # flip along 1 ax1s
        return self._topex_along_axis(flip_dem, axis=2)[:, ::-1]

    def west(self) -> np.ndarray:
        flip_dem = self.dem[:, ::-1] # flip along 1 axis
        return self._topex_along_axis(flip_dem, axis=1)[:, ::-1]

    def north_west(self) -> np.ndarray:
        flip_dem = self.dem[::-1, ::-1] # flip along 0 and 1 axes
        return self._topex_along_axis(flip_dem, axis=2)[::-1, ::-1]

    def all_directions(self) -> tuple[np.ndarray,...]:
        return (self.north(), self.north_east(), self.east(), self.south_east(),
                self.south(), self.south_west(), self.west(), self.north_west())


def run_topex_analysis(dem_path: Path, wind_dir: str,
    max_distance: float, interval: float, apply_mask: bool=False
    ) -> Union[np.ndarray,tuple[np.ndarray,...]]:

    # Read DEM file
    src = rio.open(dem_path)
    dem = src.read(1)
    res_x_y = _find_resolution(src)
    assert res_x_y, 'Image resolution was not possible to detect'

    topex = Topex(dem,
                  max_distance,
                  interval,
                  y_res=res_x_y[0],
                  x_res=res_x_y[1])

    results = np.empty(dem.shape) # Avoiding 'possibly unbound' error: Pylance
    if wind_dir == 'N':
        results = topex.north()
    if wind_dir == 'NE':
        results = topex.north_east()
    if wind_dir == 'E':
        results = topex.east()
    if wind_dir == 'SE':
        results = topex.south_east()
    if wind_dir == 'S':
        results = topex.south()
    if wind_dir == 'SW':
        results = topex.south_west()
    if wind_dir == 'W':
        results = topex.west()
    if wind_dir == 'NW':
        results = topex.north_west()
    if wind_dir == 'All':
        results = topex.all_directions()

    if apply_mask: # Apply sea mask to clean the artifacts
        land_mask = dem==0.
        sea_mask = ~land_mask
        results = results * sea_mask
        if isinstance(type(results), np.ndarray):
            results = tuple(result * sea_mask for result in results)
    return results


EARTH_RADIUS = 6_371_000  # m (Average Earth radius)
DEGREE_LENGTH = 2 * np.pi * EARTH_RADIUS / 360


def get_raster_profile(dem_path: Path) -> dict:
    src = rio.open(dem_path)
    return src.profile


def _find_resolution(src: DatasetReader) -> Optional[tuple[float,float]]:
    assert src.crs, 'Missing or invalid CRS'

    x_res, y_res = src.res

    if src.crs.is_geographic:
        R = 6_371_000  # m (Average Earth radius)
        DEGREE_LENGTH = 2 * np.pi * R / 360
        # image res in meters
        return (y_res * DEGREE_LENGTH,
                x_res * DEGREE_LENGTH * np.cos(
                    np.deg2rad(src.meta['transform'][5]))
                )
    else: # Assuming src.crs.is_projected is True
        return y_res, x_res