import os
import pickle
import numpy as np
from dataset_classes.base_preprocessing import BaseDatasetPreprocessing
from typing import Callable, Dict, Union
from preprocessing.transformations import (
    StackTransforms,
    Lambda,
    Select,
    Binarize,
    subtract_by_one,
)


#####################################################################################################
#                                      DEAP-PREPROCESSING-CLASS                                     #
#####################################################################################################
class DEAP(BaseDatasetPreprocessing):
    """
    Preprocessing Dataset class for DEAP dataset for RBTransformer.
    """

    def __init__(
        self,
        root_path: str = "./data_preprocessed_python",
        trial_window_size: int = 512,
        baseline_window_size: int = 128,
        num_channels: int = 32,
        num_baseline: int = 3,
        stride: int = 117,
        label_transform: Union[None, Callable] = None,
        num_workers: int = 0,
    ):
        super().__init__(
            root_path=root_path,
            num_channels=num_channels,
            num_baseline=num_baseline,
            trial_window_size=trial_window_size,
            baseline_window_size=baseline_window_size,
            stride=stride,
            label_transform=label_transform,
            num_workers=num_workers,
        )

    @staticmethod
    def read_record(
        record: str, root_path: str = "./data_preprocessed_python", **kwargs
    ) -> Dict:
        """
        Reads a record from the DEAP dataset and returns the samples and labels.

        Args:
            record (str): Name of the record file.
            root_path (str): Root path of the DEAP dataset.

        Returns:
            Dict: EEG samples and labels.
        """
        with open(os.path.join(root_path, record), "rb") as f:
            pkl_data = pickle.load(f, encoding="iso-8859-1")
        samples = pkl_data["data"]
        labels = pkl_data["labels"]
        return {"samples": samples, "labels": labels}

    def process_record(
        self, record: str, samples: np.ndarray, labels: np.ndarray, **kwargs
    ):
        """
        Processes EEG samples from a DEAP record and yields fixed-length segments along with corresponding labels.

        Args:
            record (str): Record identifier.
            samples (np.ndarray): Raw EEG data.
            labels (np.ndarray): Valence, arousal, dominance, and liking scores.

        Yields:
            dict: For trial segments, returns {'eeg', 'key', 'info'}.
                For the baseline segment, returns {'eeg', 'key'}.
        """
        subject_id = record
        write_pointer = 0

        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, : self.num_channels]

            trial_baseline_sample = trial_samples[
                :, : self.baseline_window_size * self.num_baseline
            ]
            trial_baseline_sample = trial_baseline_sample.reshape(
                self.num_channels, self.num_baseline, self.baseline_window_size
            ).mean(axis=1)

            trial_meta_info = {"subject_id": subject_id, "trial_id": trial_id}
            trial_rating = labels[trial_id]
            for idx, name in enumerate(["valence", "arousal", "dominance", "liking"]):
                trial_meta_info[name] = trial_rating[idx]

            start_at = self.baseline_window_size * self.num_baseline

            write_pointer = yield from self._yield_windows(
                trial_samples=trial_samples,
                trial_meta=trial_meta_info,
                write_ptr=write_pointer,
                record_prefix=record,
                start_at=start_at,
                baseline_sample=trial_baseline_sample,
            )

    def set_records(self, root_path: str = "./data_preprocessed_python", **kwargs):
        """
        Returns the list of all records in the DEAP dataset directory.

        Args:
            root_path (str): Root path of the DEAP dataset.

        Returns:
            List[str]: List of all records.
        """
        assert os.path.exists(root_path)
        return os.listdir(root_path)


if __name__ == "__main__":
    #####################################################################################################
    #                             DEAP-BINARY-VALENCE-DATASET-PREPROCESSING                             #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("valence"),
            Binarize(5.0),
        ]
    )

    deap_binary_valence_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_binary_valence_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_binary_valence_dataset, f)

    #####################################################################################################
    #                             DEAP-BINARY-AROUSAL-DATASET-PREPROCESSING                             #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("arousal"),
            Binarize(5.0),
        ]
    )

    deap_binary_arousal_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_binary_arousal_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_binary_arousal_dataset, f)

    #####################################################################################################
    #                            DEAP-BINARY-DOMINANCE-DATASET-PREPROCESSING                            #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("dominance"),
            Binarize(5.0),
        ]
    )

    deap_binary_dominance_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_binary_dominance_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_binary_dominance_dataset, f)

    #####################################################################################################
    #                              DEAP-MULTI-VALENCE-DATASET-PREPROCESSING                             #
    #####################################################################################################
    label_transform = StackTransforms([Select("valence"), Lambda(subtract_by_one)])

    deap_multi_valence_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_multi_valence_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_multi_valence_dataset, f)

    #####################################################################################################
    #                             DEAP-MULTI-AROUSAL-DATASET-PREPROCESSING                              #
    #####################################################################################################
    label_transform = StackTransforms([Select("arousal"), Lambda(subtract_by_one)])

    deap_multi_arousal_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_multi_arousal_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_multi_arousal_dataset, f)

    #####################################################################################################
    #                            DEAP-MULTI-DOMINANCE-DATASET-PREPROCESSING                             #
    #####################################################################################################
    label_transform = StackTransforms([Select("dominance"), Lambda(subtract_by_one)])

    deap_multi_dominance_dataset = DEAP(
        root_path="./data_preprocessed_python",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=32,
        num_baseline=3,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/deap_multi_dominance_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(deap_multi_dominance_dataset, f)
