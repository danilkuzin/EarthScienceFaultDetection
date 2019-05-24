import sys

from scripts.visualise_training_patches_on_map import visualise_training_patches, \
    visualise_training_patches_single_files

sys.path.extend(['../../EarthScienceFaultDetection'])

import logging
logging.basicConfig(level= logging.INFO)

#visualise_training_patches(dataset=6)

visualise_training_patches_single_files(dataset=6, num=15000)

