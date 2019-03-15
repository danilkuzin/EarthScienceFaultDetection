from src.pipeline.data_visualisation import visualise
import logging

logging.basicConfig(level=logging.DEBUG)
visualise(datasets_ind=[6], num_patches=7, patch_size=(150, 150), bands=12, plot_distributions=True,
          inp_output_path="../visualisation/")
