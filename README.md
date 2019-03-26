# Installation
* seaborn has an optional requirements statsmodels that significantly increases the speed of plotting distributions
#### GDAL
##### macos
```bash
brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/d850619e6c8dbbb29d9b2349b5b823f0548ab769/Formula/gdal.rb
gdal-config --version
<venv_path>/bin/pip install gdal==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```

##### ubuntu
```bash
sudo apt-get install libgdal-dev
sudo apt-get install python3-dev
<venv_path>/bin/pip install gdal==2.2.3 --global-option=build_ext --global-option="-I/usr/include/gdal/"
```


