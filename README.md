# Installation
* seaborn has an optional requirements statsmodels that significantly increases the speed of plotting distributions
#### GDAL
##### macos
```bash
brew install gdal
gdal-config --version
<venv_path>/bin/pip install gdal==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```

##### ubuntu
```bash
sudo apt-get install libgdal-dev
sudo apt-get install python3-dev
<venv_path>/bin/pip install gdal==2.2.3 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```


