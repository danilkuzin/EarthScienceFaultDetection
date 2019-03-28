import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5
from src.LearningKeras.train import KerasTrainer
from src.pipeline import global_params
from src.postprocessing.postprocessor import PostProcessor

trainer = KerasTrainer(model_generator=None, ensemble_size=1)

model = cnn_150x150x5()
model.load_weights('training_on_01_short_split_validation/model.h5')
trainer.models.append(model)

datasets = list(range(6))

for (preprocessor_ind, data_preprocessor_generator) in enumerate(global_params.data_preprocessor_generators):
    if preprocessor_ind not in datasets:
        continue

    data_preprocessor = data_preprocessor_generator(Mode.TEST)
    boxes, probs = trainer.apply_for_sliding_window(
        data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=25, batch_size=16,
        channels=[0, 1, 2, 3, 4])
    original_2dimage_shape = (data_preprocessor.get_data_shape()[0], data_preprocessor.get_data_shape()[1])
    faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                         original_2dimage_shape=original_2dimage_shape)
    res_faults = faults_postprocessor.heatmaps(mode="mean")
    data_preprocessor.data_io_backend.write_surface("training_on_01_short_split_validation/heatmaps_trained_on_01_short_faults_{}.tif".format(preprocessor_ind), res_faults)

    # nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
    #                                             original_2dimage_shape=original_2dimage_shape)
    # res_nonfaults = nonfaults_postprocessor.heatmaps(mode="mean")
    # data_preprocessor.data_io_backend.write_surface("training_on_01_split_validation/heatmaps_0_1_train_ratio_0.8_nonfaults_{}.tif".format(preprocessor_ind),
    #                                                     res_nonfaults)
