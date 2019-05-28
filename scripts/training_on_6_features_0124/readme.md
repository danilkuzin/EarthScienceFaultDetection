Experiment on training on region 6 - Region 7 - Nevada train
and labels manually provided for them rather than from the US database (at the moment of 25.05.2019)
For training only optical rgb and slope channels are used rather than additional elevation as in the previous experiments

The trained model is later used to build heatmaps for all available regions.

Training is done for 10 epochs.

To repeat the experiment use:
1. generate_data_6.py with:
```python
    generate_data_single_files(datasets=[6], size=15000, lookalike_ratio=[None, None, None])
```
with data augmentation as per commit 9b99029
2. run_training_on_6_split_validation.py:
```python
    model = cnn_150x150x4()

    batch_size = 32

    train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
        datasets_on_single_files(regions=[6], channels=[0, 1, 2, 4], train_ratio=0.80, batch_size=batch_size)

    train_on_preloaded_single_files(model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
                                    folder="training_on_6_features_0124", epochs=10, batch_size=batch_size)
```
3. run_predicting_on_6.py

The first point is the same for training_on_6, training_on_6_features_0124. It is required to be called once for all experiments
