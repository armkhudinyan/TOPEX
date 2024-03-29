from pathlib import Path
import numpy as np
import rasterio as rio

from .utils import find_raster_resolution


class Topex:

    def __init__(
            self,
            dem: np.ndarray,
            max_distance: float,
            interval: float,
            *,
            y_res: float,
            x_res: float | None=None
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

    def all_directions(self) -> list[np.ndarray]:
        return [self.north(), self.north_east(), self.east(), self.south_east(),
                self.south(), self.south_west(), self.west(), self.north_west()]


def run_topex_analysis(
        dem_path: Path,
        wind_dir: str,
        max_distance: float,
        interval: float,
        apply_mask: bool=False
    ) -> np.ndarray | list[np.ndarray]:

    # Read DEM file
    src = rio.open(dem_path)
    dem = src.read(1)
    res_y_x = find_raster_resolution(src)
    assert res_y_x, 'Image resolution was not possible to detect'

    topex = Topex(dem,
                  max_distance,
                  interval,
                  y_res=res_y_x[0],
                  x_res=res_y_x[1])

    topex_map = np.empty(dem.shape) # Avoiding 'possibly unbound' error: Pylance
    if wind_dir == 'N':
        topex_map = topex.north()
    if wind_dir == 'NE':
        topex_map = topex.north_east()
    if wind_dir == 'E':
        topex_map = topex.east()
    if wind_dir == 'SE':
        topex_map = topex.south_east()
    if wind_dir == 'S':
        topex_map = topex.south()
    if wind_dir == 'SW':
        topex_map = topex.south_west()
    if wind_dir == 'W':
        topex_map = topex.west()
    if wind_dir == 'NW':
        topex_map = topex.north_west()
    if wind_dir == 'All':
        topex_map = topex.all_directions()

    if apply_mask: # Apply sea mask to clean the artifacts
        land_mask = dem==0.
        sea_mask = ~land_mask
        topex_map = topex_map * sea_mask
        if isinstance(type(topex_map), np.ndarray):
            topex_map = list(result * sea_mask for result in topex_map)
    return topex_map