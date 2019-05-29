import io
import pathlib

import h5py
import yaml

from src.config import data_preprocessor_params, data_path
import numpy as np


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