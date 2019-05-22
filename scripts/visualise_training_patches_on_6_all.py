import sys

from scripts.visualise_training_patches_on_map import visualise_training_patches

sys.path.extend(['../../EarthScienceFaultDetection'])

import logging
logging.basicConfig(level= logging.INFO)

visualise_training_patches(dataset=6)

