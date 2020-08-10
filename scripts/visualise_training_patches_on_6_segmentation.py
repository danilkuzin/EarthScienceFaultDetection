import logging
import sys

from scripts.visualise_training_patches_on_map import \
    visualise_training_patches_single_files_segmentation

sys.path.extend(['../../EarthScienceFaultDetection'])

logging.basicConfig(level=logging.INFO)

visualise_training_patches_single_files_segmentation(dataset=6, num=1654)
