Experiment on training on region 6 - Nevada train and building heatmaps for all available regions. Training is done for 10 epochs.

To repeat the experiment use:
1. generate_data.py with:
```python
    datasets = [6]
```
and
```python
    imgs, lbls = data_generator.create_datasets(
    class_probabilities="two-class",
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    size=2000)
```
2. run_training_on_6_split_validation.py
3. run_predicting_on_6_split_validation.py