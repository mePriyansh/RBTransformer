import os
import pickle
import numpy as np
import scipy.io as scio
from dataset_classes.base_preprocessing import BaseDatasetPreprocessing
from typing import Callable, Dict, Union
from preprocessing.transformations import StackTransforms, Lambda, Select, add_by_one


class SEED(BaseDatasetPreprocessing):
    """Preprocessing Dataset class for SEED dataset for RBTransformer."""

    def __init__(
        self,
        root_path: str = "./Preprocessed_EEG",
        num_channels: int = 62,
        trial_window_size: int = 512,
        stride: int = 117,
        num_baseline: Union[int, None] = None,
        baseline_window_size: Union[int, None] = None,
        label_transform: Union[None, Callable] = None,
        num_workers: int = 0,
    ):
        super().__init__(
            root_path=root_path,
            num_channels=num_channels,
            trial_window_size=trial_window_size,
            stride=stride,
            num_baseline=num_baseline,
            baseline_chunk_size=baseline_window_size,
            label_transform=label_transform,
            num_workers=num_workers,
        )

    @staticmethod
    def read_record(
        record: str, root_path: str = "./Preprocessed_EEG", **kwargs
    ) -> Dict:
        """
        Reads a record from the SEED dataset and returns the samples and labels.

        Args:
            record (str): Name of the record file.
            root_path (str): Root path of the SEED dataset.

        Returns:
            Dict: EEG samples and labels.
        """
        samples = scio.loadmat(
            os.path.join(root_path, record),
            verify_compressed_data_integrity=False,
        )
        labels = scio.loadmat(
            os.path.join(root_path, "label.mat"),
            verify_compressed_data_integrity=False,
        )["label"][0]
        return {"samples": samples, "labels": labels}

    def process_record(self, record: str, samples: Dict, labels: np.ndarray, **kwargs):
        """
        Processes EEG samples from a SEED record and yields fixed-length segments along with corresponding labels.

        Args:
            record (str): Filename of the .mat EEG record (e.g., 1_20131027.mat).
            samples (Dict): Loaded EEG signal data for the subject-session.
            labels (np.ndarray): Emotion labels for each trial.

        Yields:
            dict: For trial segments, returns {'eeg', 'key', 'info'}.
                For the baseline segment, returns {'eeg', 'key'} (not used in SEED).
        """
        subject_id = int(os.path.basename(record).split(".")[0].split("_")[0])
        session_date = int(os.path.basename(record).split(".")[0].split("_")[1])
        write_pointer = 0

        trial_ids = [key for key in samples.keys() if "eeg" in key]

        for trial_id in trial_ids:
            trial_samples = samples[trial_id][: self.num_channels]

            trial_meta_info = {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "emotion": int(labels[int(trial_id.split("_")[-1][3:]) - 1]),
                "date": session_date,
            }

            write_pointer = yield from self._yield_windows(
                trial_samples=trial_samples,
                trial_meta=trial_meta_info,
                write_ptr=write_pointer,
                record_prefix=record,
                start_at=0,
                baseline_sample=None,
            )

    def set_records(self, root_path: str = "./Preprocessed_EEG", **kwargs):
        """
        Returns the list of all records in the SEED dataset directory.

        Args:
            root_path (str): Root path of the SEED dataset.

        Returns:
            List[str]: List of all records.
        """
        assert os.path.exists(root_path)
        file_list = os.listdir(root_path)
        skip_set = ["label.mat", "readme.txt"]
        return [f for f in file_list if f not in skip_set]


if __name__ == "__main__":
    #####################################################################################################
    #                                  SEED-MULTI-DATASET-PREPROCESSING                                 #
    #####################################################################################################
    label_transform = StackTransforms([Select("emotion"), Lambda(add_by_one)])

    seed_multi_dataset = SEED(
        root_path="./Preprocessed_EEG",
        trial_window_size=512,
        num_channels=62,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/seed_multi_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(seed_multi_dataset, f)
