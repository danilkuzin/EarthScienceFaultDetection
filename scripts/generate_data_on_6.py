import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.generate_data import generate_data

generate_data(datasets=[6], size=2000)
