import numpy as np
import abc


class DataIOBackend:
    @abc.abstractmethod
    def load_elevation(self, path: str) -> np.array:
        """elevation map (from the Shuttle Radar Topography Mission), values in meters above sea level"""
        pass

    @abc.abstractmethod
    def load_slope(self, path: str) -> np.array:
        """slope map derived from the elevation, values in degrees from horizontal, 0-90"""
        pass

    @abc.abstractmethod
    def load_optical(self, path_r: str, path_g: str, path_b:str) -> np.array:
        """standard red / green / blue optical bands from the Landsat-8 platform, each 0 - 255"""
        pass

    @abc.abstractmethod
    def load_features(self, path: str) -> np.array:
        """0 - neutral, undefined content (could include faults--fair area for testing)
           1 - faults
           2 - fault lookalikes - features that we think share visual or topographic similarity with faults, but expert interpretation can exclude
           3 - not-faults - areas that definitely do not include faults, nor things that we think even look like faults, can be used directly for training what faults are not.
        """
        pass

    @abc.abstractmethod
    def load_nir(self, path: str) -> np.array:
        """near infrared from Landsat"""
        pass

    @abc.abstractmethod
    def load_ir(self, path: str) -> np.array:
        """infrared from Landsat"""
        pass

    @abc.abstractmethod
    def load_swir1(self, path: str) -> np.array:
        """shortwave infrared1 from Landsat"""
        pass

    @abc.abstractmethod
    def load_swir2(self, path: str) -> np.array:
        """shortwave infrared2 from Landsat"""
        pass

    @abc.abstractmethod
    def load_panchromatic(self, path: str) -> np.array:
        """panchromatic band from Landsat, essentially just total surface reflectance,
        like a grayscale image of the ground"""
        pass
