import argparse

from src.pipeline.training import train

parser = argparse.ArgumentParser()
parser.add_argument('train_datasets_numbers', type=int, nargs='+', choices=[1, 2])

args = parser.parse_args()
print("training on datasets: {}".format(args.train_datasets_numbers))

# train(
#     train_datasets=args.train_datasets_numbers,
#     class_probabilities="equal",
#     batch_size=10,
#     patch_size=(150, 150),
#     channels=[0, 1, 2, 3, 4],
#     ensemble_size=2
# )

# train(
#     train_datasets=args.train_datasets_numbers,
#     class_probabilities="two-class",
#     batch_size=10,
#     patch_size=(150, 150),
#     channels=[0, 1, 2, 3, 4],
#     ensemble_size=2
# )

train(
    train_datasets=args.train_datasets_numbers,
    class_probabilities="two-class",
    batch_size=10,
    patch_size=(150, 150),
    channels=[4],
    ensemble_size=2
)
