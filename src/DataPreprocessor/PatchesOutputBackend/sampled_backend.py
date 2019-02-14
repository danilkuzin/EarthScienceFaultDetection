from enum import Enum

from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend


class DatasetType(Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3


class SampledBackend(PatchesOutputBackend):
    def __init__(self):
        self.dirs = dict()
        self.datasets_sizes = dict()
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.FAULT.name] = 100
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.NONFAULT.name] = 100
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.FAULT.name] = 20
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.NONFAULT.name] = 20
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.FAULT.name] = 10
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.NONFAULT.name] = 10
        self.prepare_folders()

    def prepare_folders(self):
        if self.mode == Mode.TRAIN:
            self.dirs['train_fault'] = self.data_dir + "learn/train/fault/"
            self.dirs['train_nonfault'] = self.data_dir + "learn/train/nonfault/"
            self.dirs['valid_fault'] = self.data_dir + "learn/valid/fault/"
            self.dirs['valid_nonfault'] = self.data_dir + "learn/valid/nonfault/"
            self.dirs['test_w_labels_fault'] = self.data_dir + "learn/test_with_labels/fault/"
            self.dirs['test_w_labels_nonfault'] = self.data_dir + "learn/test_with_labels/nonfault/"
            self.dirs['test'] = self.data_dir + "learn/test/test/"
            pathlib.Path(self.dirs['train_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['train_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['valid_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['valid_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test_w_labels_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test_w_labels_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test']).mkdir(parents=True, exist_ok=True)
        self.dirs['all_patches'] = self.data_dir + "all/"
        pathlib.Path(self.dirs['all_patches']).mkdir(parents=True, exist_ok=True)

    def prepare_datasets(self, output_backend, patch_size):
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT_LOOKALIKE, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT_LOOKALIKE, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT_LOOKALIKE, patch_size)

    def prepare_dataset(self, output_backend: PatchesOutputBackend, data_type, label, patch_size):
        category = data_type.name + "_" + label.name
        arr = np.zeros(
            (self.datasets_sizes[category], patch_size[0], patch_size[1], self.num_channels))
        for i in trange(self.datasets_sizes[category]):
            arr[i] = self.sample_patch(label)
        output_backend.save(arr, label==1 if 0 else 1, self.dirs[category])

    def prepare_all_patches(self, backend: PatchesOutputBackend, patch_size):
        for i, j in tqdm(itertools.product(range(self.optical_rgb.shape[0] // patch_size[0]),
                        range(self.optical_rgb.shape[1] // patch_size[1]))):
            left_border = i * patch_size[0]
            right_border = (i + 1) * patch_size[0]
            top_border = j * patch_size[0]
            bottom_border = (j + 1) * patch_size[0]
            cur_patch = self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)
            backend.save(array=cur_patch, label=0, path=self.dirs['all_patches'] + "/{}_{}.tif".format(i, j))
