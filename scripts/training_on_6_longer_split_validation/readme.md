Experiment on training on region 6 - Nevada train. Similar to training_on_6_split_validation, but training is done for larger number of epochs (30 rather than 10)

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
2. run_training_on_6_longer_split_validation.py

Results show that on 30 epochs there is an overfitting when accuracy on train continue to grow and accuracy on validation drops.
