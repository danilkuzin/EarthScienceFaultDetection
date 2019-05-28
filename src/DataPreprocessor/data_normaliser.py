import h5py
import yaml

from src.config import data_preprocessor_params, areas, data_path
import numpy as np

import pathlib
import io


class RegionNormaliser:
    def __init__(self, region_id, area_ind):
        self.region_id = region_id
        self.area_ind = area_ind
        self.elevation = None
        self.slope = None
        self.optical_r = None
        self.optical_g = None
        self.optical_b = None
        self.features = None
        self.elevation_normalised = None
        self.slope_normalised = None
        self.optical_r_normalised = None
        self.optical_g_normalised = None
        self.optical_b_normalised = None

    def load(self):
        preprocessed_data_path = f"{data_path}/preprocessed/{self.region_id}/data.h5"
        with h5py.File(preprocessed_data_path, 'r') as hf:
            self.optical_r = hf["optical_r"][:]
            self.optical_g = hf["optical_g"][:]
            self.optical_b = hf["optical_b"][:]
            self.elevation = hf["elevation"][:]
            self.slope = hf["slope"][:]
            self.features = hf["features"][:]

    def normalise(self):
        optical_mean = 127.5
        optical_std = 255.
        self.optical_r_normalised = (self.optical_r - optical_mean) / optical_std
        self.optical_g_normalised = (self.optical_g - optical_mean) / optical_std
        self.optical_b_normalised = (self.optical_b - optical_mean) / optical_std

        input_path = f"{data_path}/normalised"
        with open(f"{input_path}/area_{self.area_ind}.yaml", 'r') as stream:
            features_areawide = yaml.safe_load(stream)

        elevation_mean, elevation_std = features_areawide['mean_elevation'], features_areawide['std_elevation']
        self.elevation_normalised = (self.elevation - elevation_mean) / elevation_std

        slope_mean = 45.
        slope_std = 90.
        self.slope_normalised = (self.slope - slope_mean) / slope_std

    def save_results(self):
        with h5py.File(f"{data_path}/normalised/{self.region_id}_data.h5", 'w') as hf:
            hf.create_dataset("elevation", data=self.elevation_normalised)
            hf.create_dataset("slope", data=self.slope_normalised)
            hf.create_dataset("optical_r", data=self.optical_r_normalised)
            hf.create_dataset("optical_g", data=self.optical_g_normalised)
            hf.create_dataset("optical_b", data=self.optical_b_normalised)
            hf.create_dataset("features", data=self.features)


class AreaNormaliser:
    def __init__(self, area_ind):
        self.area_ind = area_ind

    def normalise(self):
        associated_regions = []
        for region_id, region in enumerate(data_preprocessor_params):
            if region[1] == self.area_ind:
                associated_regions.append(region_id)

        print(f"area: {self.area_ind}, associated regions: {associated_regions}")
        elevations_list = []
        for region_id in associated_regions:
            preprocessed_data_path = f"{data_path}/preprocessed/{region_id}/data.h5"
            with h5py.File(preprocessed_data_path, 'r') as hf:
                elevations_list.append(hf["elevation"][:])

        for i in range(len(elevations_list)):
            elevations_list[i] = elevations_list[i].flatten()

        elevations_array = np.concatenate(elevations_list)
        mean_elevation = np.mean(elevations_array)
        std_elevation = np.std(elevations_array)

        output_path = f"{data_path}/normalised"
        pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)

        with io.open(f"{output_path}/area_{self.area_ind}.yaml", 'w', encoding='utf8') as outfile:
            yaml.dump({"mean_elevation": float(mean_elevation), "std_elevation": float(std_elevation)}, outfile,
                      default_flow_style=False, allow_unicode=True)