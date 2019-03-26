import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.data_visualisation import visualise
import logging

logging.basicConfig(level=logging.DEBUG)
visualise(datasets_ind=[0], num_patches=0, patch_size=(150, 150), bands=12, plot_distributions=True,
          inp_output_path="../report/graphics/data/", crop=(500, 1400, 1000, 1900))
