import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.data_visualisation import visualise
import logging

logging.basicConfig(level=logging.DEBUG)
visualise(datasets_ind=[0, 1, 2, 3, 4, 5, 6, 7], num_patches=7, patch_size=(150, 150), bands=12, plot_distributions=True,
          inp_output_path="../visualisation/")
