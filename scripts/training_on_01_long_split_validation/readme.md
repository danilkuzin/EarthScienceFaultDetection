Experiment on training on region 0 and 1 - Tibet Region 1 - Lopukangri and Region 2 - Muga Puruo - and building heatmaps for all available regions. 

Training is done for 10 epochs.

To repeat the experiment use:
1. generate_data.py with:
```python
    datasets = [0]
```
and
```python
    imgs, lbls = data_generator.create_datasets(
    class_probabilities="two-class",
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    size=2000)
```
2. generate_data.py with:
```python
    datasets = [1]
```
and
```python
    imgs, lbls = data_generator.create_datasets(
    class_probabilities="two-class",
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    size=2000)
```
2. run_training_on_01_long_split_validation.py
3. run_predicting_on_01_long_split_validation.py

The first two points are the same for training_on_01_short_split_validation, training_on_01_long_split_validation and training_on_01_longer_split_validation. They are required to be called once for all three experiments

