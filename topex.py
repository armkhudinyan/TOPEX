import numpy as np


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
        self.pixel_per_orth_interval = self.interval / self.resolution
        self.pixel_per_diag_interval = self.pixel_per_orth_interval / np.sqrt(2)

    def _distances(self, pixels_per_inteval: float) -> np.ndarray:
        '''List of distances to calculate the topex for each location '''
        return (np.arange(1, round(self.max_distance / self.interval) + 1
            ) * pixels_per_inteval).round().astype(int)

    def _topex_along_axis(self, dem: np.ndarray, axis: int=0) -> np.ndarray:
        # NOTE: # Axis 3 here refers to any of the diagonal directions

        distances = self._distances(self.pixel_per_orth_interval
                        if axis < 2 else self.pixel_per_diag_interval)
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

    def all_dir_multip(self) -> list[np.ndarray]:
        from pathos.multiprocessing import Pool

        functions = [self.north, self.north_east, self.east, self.south_east,
                     self.south, self.south_west, self.west, self.north_west]

        with Pool(processes=len(functions)) as pool:
            result = pool.map(lambda f: f(), functions)

        return result