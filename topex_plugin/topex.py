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
            resolution: float
            ) -> None:
        self.dem = dem
        self.max_distance = max_distance
        self.interval = interval
        self.resolution = resolution
        self.pixels_per_orth_interval = self.interval / self.resolution
        self.pixels_per_diag_interval = self.pixels_per_orth_interval / np.sqrt(2)

    def _distances(self, pixels_per_inteval: float) -> np.ndarray:
        '''List of distances to calculate the topex for each location '''
        return (np.arange(1, round(self.max_distance / self.interval) + 1
            ) * pixels_per_inteval).round().astype(int)

    def _topex_along_axis(self, dem: np.ndarray, axis: int=0) -> np.ndarray:
        # NOTE: # Axis 2 here refers to any of the diagonal directions

        distances = self._distances(self.pixels_per_orth_interval
                        if axis < 2 else self.pixels_per_diag_interval)
        resolution = (self.resolution
                        if axis < 2 else self.resolution * np.sqrt(2))

        pad_dist = distances[-1]
        padded_dem = np.pad(dem, pad_dist)
        slope = np.full(dem.shape, np.pi / 2)
        height, width = dem.shape

        for distance in distances:
            if axis == 0:
                delta_height = dem - padded_dem[pad_dist + distance: pad_dist +
                                                distance + height,
                                                pad_dist: pad_dist + width]
            elif axis == 1:
                delta_height = dem - padded_dem[pad_dist: pad_dist + height,
                                                pad_dist + distance: pad_dist +
                                                distance + width]
            else:
                delta_height = dem - padded_dem[pad_dist + distance: pad_dist +
                                                distance + height,
                                                pad_dist + distance: pad_dist +
                                                distance + width]
            angle = np.arctan(delta_height / (distance * resolution))
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

    # def all_dir_multip(self) -> list[np.ndarray]:
    #     from pathos.multiprocessing import Pool
    #     functions = [self.north, self.north_east, self.east, self.south_east,
    #                  self.south, self.south_west, self.west, self.north_west]
    #     with Pool(processes=len(functions)) as pool:
    #         result = pool.map(lambda f: f(), functions)

        return result


def run_topex_analysis(dem_path: Path, wind_dir: str,
    max_distance: float, interval: float, apply_mask: bool=False
    ) -> np.ndarray | tuple[np.ndarray,...]:

    # Read DEM file
    src = rio.open(dem_path)
    dem = src.read(1)
    resolution = _find_resolution(src)
    assert resolution, 'Image resolution was not possible to detect'

    topex = Topex(dem, max_distance, interval, resolution)

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


def _find_resolution(src: DatasetReader) -> float | None:
    assert src.crs, 'Missing CRS'

    if src.crs.is_geographic:
        # image res in meters
        return abs(src.meta['transform'][4]) * DEGREE_LENGTH
    elif src.crs.is_projected:
        return abs(src.meta['transform'][4])