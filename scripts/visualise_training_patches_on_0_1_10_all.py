import sys

from scripts.visualise_training_patches_on_map import visualise_training_patches, \
    visualise_training_patches_single_files

sys.path.extend(['../../EarthScienceFaultDetection'])

import logging
logging.basicConfig(level=logging.INFO)

# visualise_training_patches(dataset=0)
# visualise_training_patches(dataset=1)
# visualise_training_patches(dataset=10)

visualise_training_patches_single_files(dataset=0, num=5000)
visualise_training_patches_single_files(dataset=1, num=5000)
visualise_training_patches_single_files(dataset=10, num=5000)

