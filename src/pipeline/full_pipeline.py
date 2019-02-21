
class Pipeline:
    def __init__(self):
        self.model_dir = model_dir
        self.channels = channels
        self.patch_size = (150, 150),
        self.ensemble_size = 2,

    def run(self):
        self.create_generators()

        self.train(
            train_datasets=args.train_datasets_numbers,
            class_probabilities="two-class",
            batch_size=10,
            patch_size=(150, 150),
            channels=[5],
            ensemble_size=2,
            output_path="feature_nir/"
        )

        self.predict(models_folder="feature_nir/trained_models_12", ensemble_size=2, classes=2, channels=[5],
                heatmap_mode="mean_median", stride=50, batch_size=100)

    def create_generators(self):
        self.train_preprocessors = []
        self.train_preprocessors.append(data_preprocessor_generators_train)
        if 1 in train_datasets:
            preprocessors.append(DataPreprocessor(
                data_dir="../data/Region 1 - Lopukangri/",
                data_io_backend=GdalBackend(),
                patches_output_backend=InMemoryBackend(),
                filename_prefix="tibet",
                mode=Mode.TRAIN,
                seed=1)
            )

        if 2 in train_datasets:
            preprocessors.append(DataPreprocessor(
                data_dir="../data/Region 2 - Muga Puruo/",
                data_io_backend=GdalBackend(),
                patches_output_backend=InMemoryBackend(),
                filename_prefix="mpgr",
                mode=Mode.TRAIN,
                seed=1)
            )

        self.train_data_generator = DataGenerator(preprocessors=preprocessors)

    def train(self, train_datasets: List[int], class_probabilities: str, batch_size: int, patch_size: Tuple[int, int],
          channels: List[int], ensemble_size: int, train_lib="keras", output_path=""):
        np.random.seed(1)
        tf.set_random_seed(2)



        if class_probabilities == "equal":
            class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
            joint_generator = data_generator.generator_3class(batch_size=batch_size,
                                                       class_probabilities=class_probabilities_int,
                                                       patch_size=patch_size,
                                                       channels=np.array(channels))
            if train_lib == "keras":
                trainer = KerasTrainer(model_generator=lambda: cnn_150x150x5_3class(),
                                       ensemble_size=ensemble_size)
            elif train_lib == "tensorflow":
                trainer = KerasTrainer(model_generator=lambda: cnn_150x150x5_3class(),
                                       ensemble_size=ensemble_size)
        elif class_probabilities == "two-class":
            class_probabilities_int = np.array([0.5, 0.25, 0.25])
            joint_generator = data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=batch_size,
                                                       class_probabilities=class_probabilities_int,
                                                       patch_size=patch_size,
                                                       channels=np.array(channels))
            if len(channels) == 5:
                model_generator = lambda: cnn_150x150x5()
            elif len(channels) == 3:
                model_generator = lambda: cnn_150x150x3()
            elif len(channels) == 1:
                model_generator = lambda: cnn_150x150x1()
            else:
                raise Exception()
            trainer = KerasTrainer(model_generator=model_generator,
                                   ensemble_size=ensemble_size)
        else:
            raise Exception('Not implemented')

        history_arr = trainer.train(steps_per_epoch=50, epochs=5, train_generator=joint_generator)

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        trainer.save(output_path='{}trained_models_{}'.format(output_path, ''.join(str(i) for i in train_datasets)))

        for (hist_ind, history) in enumerate(history_arr):
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig("Model accuracy_{}.png".format(hist_ind))
            plt.close()

            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig("Model loss_{}.png".format(hist_ind))
            plt.close()

    def predict(models_folder, ensemble_size, classes, channels, heatmap_mode="max", stride=50, batch_size=20):

        if classes == 3:
            model_generator = lambda: cnn_150x150x5_3class()
        elif classes == 2:
            if len(channels) == 5:
                model_generator = lambda: cnn_150x150x5()
            elif len(channels) == 3:
                model_generator = lambda: cnn_150x150x3()
            elif len(channels) == 1:
                model_generator = lambda: cnn_150x150x1()
            else:
                raise Exception()
        else:
            raise Exception('not supported')

        trainer = KerasTrainer(model_generator=model_generator,
                               ensemble_size=ensemble_size)

        trainer.load(input_path=models_folder)

        for (preprocessor_ind, data_preprocessor_generator) in enumerate(
                global_params.data_preprocessor_generators_test):
            data_preprocessor = data_preprocessor_generator()
            boxes, probs = trainer.apply_for_sliding_window(
                data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=stride, batch_size=batch_size,
                channels=channels)
            original_2dimage_shape = (data_preprocessor.get_data_shape()[0], data_preprocessor.get_data_shape()[1])
            faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                                 original_2dimage_shape=original_2dimage_shape)
            res_faults = faults_postprocessor.heatmaps(mode=heatmap_mode)
            data_preprocessor.data_io_backend.write_surface("heatmaps_faults_{}.tif".format(preprocessor_ind),
                                                            res_faults)

            if classes == 3:
                lookalikes_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                         original_2dimage_shape=original_2dimage_shape)
                res_lookalikes = lookalikes_postprocessor.heatmaps(mode=heatmap_mode)
                data_preprocessor.data_io_backend.write_surface("heatmaps_lookalikes_{}.tif".format(preprocessor_ind),
                                                                res_lookalikes)

                nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 2],
                                                        original_2dimage_shape=original_2dimage_shape)
                res_nonfaults = nonfaults_postprocessor.heatmaps(mode="max")
                data_preprocessor.data_io_backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                                res_nonfaults)

            elif classes == 2:
                nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                        original_2dimage_shape=original_2dimage_shape)
                res_nonfaults = nonfaults_postprocessor.heatmaps(mode=heatmap_mode)
                data_preprocessor.data_io_backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                                res_nonfaults)

