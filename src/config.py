try:
    import google.colab
    from google.colab import drive
    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False

if IN_COLAB:
    data_path = '/gdrive/My Drive/Work/fault_detection/data'
    drive.mount('/gdrive')
else:
    data_path = "/mnt/data/datasets/DataForEarthScienceFaultDetection" # "../../DataForEarthScienceFaultDetection" #
areas = {"Tibet": 0, "Nevada": 1, "Turkey": 2, "California": 3}
# path, area, trainable
data_preprocessor_params = [
    (f"{data_path}/raw_data/Region 1 - Lopukangri/", 0, True),
    (f"{data_path}/raw_data/Region 2 - Muga Puruo/", 0, True),
    (f"{data_path}/raw_data/Region 3 - Muggarboibo/", 0, False),
    (f"{data_path}/raw_data/Region 4 - Austin-Tonopah/", 1, False),
    (f"{data_path}/raw_data/Region 5 - Las Vegas Nevada/", 1, False),
    (f"{data_path}/raw_data/Region 6 - Izmir Turkey/", 2, False),
    (f"{data_path}/raw_data/Region 7 - Nevada train/", 1, True),
    (f"{data_path}/raw_data/Region 8 - Nevada test/", 1, True),
    (f"{data_path}/raw_data/Region 8 - 144036/", 0, False),
    (f"{data_path}/raw_data/Region 9 - WRS 143038/", 0, False),
    (f"{data_path}/raw_data/Region 10 - 141037/", 0, True),
    (f"{data_path}/raw_data/Region 11 - 140038/", 0, False),
    (f"{data_path}/raw_data/Region 12 - Central California", 3, True)
]


