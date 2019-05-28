Experiment on training on region 0, 1, and 10 - Tibet Region 1 - Lopukangri and Region 2 - Muga Puruo and Region 10 - 141037
and all labels available for them (at the moment of 25.05.2019)

The trained model is later used to build heatmaps for all available regions.

The CNN model in this experiment is cnn_four_layers with four convolutional layers instead of two in cnn_150x150x5

Training is done for 10 epochs initially, continued for another 20 epochs but stopped by early stopping at 2nd epoch (12th in total).

To repeat the experiment use:
1. generate_data_0_1_10.py with:
```python
    generate_data_single_files(datasets=[0, 1, 10], size=5000, lookalike_ratio=[None, None, 0.01])
```
with data augmentation as per commit 9b99029
2. run_training_on_0_1_10_deeper.py:
```python
    model = cnn_four_layers()

    batch_size = 32

    train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
        datasets_on_single_files(regions=[0, 1, 10], channels=[0, 1, 2, 3, 4], train_ratio=0.80, batch_size=batch_size)

    train_on_preloaded_single_files(model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
                                    folder="training_on_0_1_10_deeper", epochs=10, batch_size=batch_size)
```
3. run_training_on_0_1_10_deeper_continue.py:
```python
    model = cnn_four_layers()

    model.load_weights('training_on_0_1_10_deeper/model_10_epoch.h5')

    batch_size = 32

    train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
        datasets_on_single_files(regions=[0, 1, 10], channels=[0, 1, 2, 3, 4], train_ratio=0.80, batch_size=batch_size)

    train_on_preloaded_single_files(model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
                                    folder="training_on_0_1_10_deeper", epochs=20, batch_size=batch_size)
```
3. run_predicting_on_0_1_10_deeper.py

The first point is the same for training_on_0_1_10, training_on_0_1_10_deeper. It is required to be called once for all experiments
