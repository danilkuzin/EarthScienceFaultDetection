import numpy as np
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x3, cnn_150x150x5
from src.pipeline import global_params

np.random.seed(1)
tf.set_random_seed(2)

train_data_preprocessor = global_params.data_preprocessor_generators[0](Mode.TRAIN)
test_data_preprocessor = global_params.data_preprocessor_generators[1](Mode.TRAIN)

test_size = 500

img_test, lbl_test = test_data_preprocessor.get_sample_2class_lookalikes_with_nonfaults(batch_size=test_size,
                                                                                          class_probabilities=np.array([0.5, 0.25, 0.25]),
                                                                                          patch_size=(150, 150),
                                                                                          channels=np.array([0, 1, 2, 3, 4]))
test_lbls = np.argmax(lbl_test, axis=1)

quality_test = []
model = cnn_150x150x5()

train_generator = train_data_preprocessor.train_generator_2class_lookalikes_with_nonfaults(batch_size=50,
                                                                                           class_probabilities=np.array([0.5, 0.25, 0.25]),
                                                                                           patch_size=(150, 150),
                                                                                           channels=np.array([0, 1, 2, 3, 4]))
num_epochs = 50
for ep in range(num_epochs):
    model.fit_generator(generator=train_generator,
                              steps_per_epoch=10,
                              epochs=1,
                              validation_data=train_generator,
                              validation_steps=5,
                              workers=0,
                              use_multiprocessing=False)
    pred_lbls = np.argmax(model.predict(img_test), axis=1)

    quality_test.append(np.sum(pred_lbls == test_lbls)/lbl_test.shape[0])

    print(quality_test)

