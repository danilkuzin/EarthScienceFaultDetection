# Installation
* seaborn has an optional requirements statsmodels that significantly increases the speed of plotting distributions
#### GDAL
The current version of gdal in brew does not match the one in pip, so we install older version from brew. Once it changes, the brew version can be used
##### macos
```bash
brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/d850619e6c8dbbb29d9b2349b5b823f0548ab769/Formula/gdal.rb
gdal-config --version
<venv_path>/bin/pip install gdal==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```

##### mac via conda
there were some gcc problems on mac the following worked:
```bash
conda install gdal
```

##### ubuntu
```bash
sudo apt-get install libgdal-dev
sudo apt-get install python3-dev
<venv_path>/bin/pip install gdal==2.2.3 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```

#### Other python libraries
```bash
pip install pillow statsmodels seaborn tensorflow tqdm opencv-python h5py
```

# Repeating experiments from report
* get data from GDrive, links are not posted for security reasons. Unarchive it and place it in DataForEarthScienceFaultDetection/raw_data
* scripts/run_preprocessing.py
* scripts/generate_data_on_0_1_10.py
* scripts/generate_data_on_6.py
* scripts/run_training_on_0_1__10.py
* scripts/training_on_6_split_validation.py
* scripts/run_predicting_on_0_1_10.py - you may want to increase stride to 50 to make it faster
* scripts/run_predicting_on_6.py - same comment here

### optional
* scripts/run_visualisation_for_report.py (or use scripts/run_visualisation.py to get much more images) - all 
visualisations may take few gigabytes of space
* scripts/run_feature_selection.py - long and memory leaks
* scripts/run_nn_visualisation.py


# Code organisation
* scripts - high level scripts to get everything running
* src/pipeline/global_params - description of the datasets
* src/DataPreprocessor - handling the data: loading, checking, normalising, visualising also names of the channels are here, sampling.
* src/DataPreprocessor/DataIOBackend/gdal_backend - scripts for handling GeoTiffs
* src/LearningKeras/ - two versions of fitting models, non-2 is used, 2 can be used for finer control. 
Architectures are described there as well.
* src/pipeline - high level functions to operate with data. scripts from /scripts folder are usually using this folder
* src/postprocessing - heatmaps, etc


# Common problems
on MAC add the following line to scripts to avoid the python not installed as a framework problem:
```python
import matplotlib
matplotlib.use('tkagg')
```

if running not from pycharm, but from command line add the following to avoid the problem of modules not found:
```python
import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

```

# Notes
* Some of fault lines provided for Nevada-train (region 6) have UTM coordinates outside of region boundaries. These fault lines are ignored in the current code. 
* For first two regions tif for different bands have slightly different size (couple of pixels difference). Larger tifs are cropped to fit the smallest band for the same region.

# Related
* Related application of cancerous cell detection from DeepSets with Attention by Ilse Tomczak Welling 2018
