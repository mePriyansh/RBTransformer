import os
import pickle
import numpy as np
import mne
from dataset_classes.base_preprocessing import BaseDatasetPreprocessing
from typing import Callable, Dict, Union
from preprocessing.transformations import (
    StackTransforms,
    Normalize,
    BandDifferentialEntropy,
    SubtractBaseline,
    UnsqueezeDim,
    Lambda,
    Select,
)


MI_RUNS_LEFT_RIGHT = [3, 7, 11]
MI_RUNS_HANDS_FEET = [4, 8, 12]
MI_RUNS = MI_RUNS_LEFT_RIGHT + MI_RUNS_HANDS_FEET

EPOCH_SAMPLES = 656


class PhysioNetMI(BaseDatasetPreprocessing):
    """
    Preprocessing Dataset class for PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).
    4-class motor imagery: left fist, right fist, both fists, both feet.
    """

    def __init__(
        self,
        root_path: str = "./physionet_mi_data/physionet.org/files/eegmmidb/1.0.0",
        trial_window_size: int = 256,
        num_channels: int = 64,
        stride: int = 32,
        label_transform: Union[None, Callable] = None,
        num_workers: int = 8,
    ):
        super().__init__(
            root_path=root_path,
            num_channels=num_channels,
            trial_window_size=trial_window_size,
            stride=stride,
            num_baseline=None,
            baseline_window_size=None,
            label_transform=label_transform,
            num_workers=num_workers,
        )

        self.preprocessing_transformations = StackTransforms(
            [
                Normalize(),
                BandDifferentialEntropy(sampling_rate=160),
                SubtractBaseline(),
                UnsqueezeDim(),
            ]
        )

    @staticmethod
    def read_record(
        record: str,
        root_path: str = "./physionet_mi_data/physionet.org/files/eegmmidb/1.0.0",
        **kwargs,
    ) -> Dict:
        """
        Reads all 6 motor imagery runs for a given subject.

        Args:
            record (str): Subject folder name (e.g., "S001").
            root_path (str): Root path of the EEGMMIDB dataset.

        Returns:
            Dict: Contains lists of raw data arrays, events, and run numbers.
        """
        subject_dir = os.path.join(root_path, record)
        raw_arrays = []
        all_events = []
        run_numbers = []

        for run_num in MI_RUNS:
            edf_file = os.path.join(subject_dir, f"{record}R{run_num:02d}.edf")
            if not os.path.exists(edf_file):
                continue

            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            data = raw.get_data()
            events, _ = mne.events_from_annotations(raw, verbose=False)

            raw_arrays.append(data)
            all_events.append(events)
            run_numbers.append(run_num)

        return {
            "raw_arrays": raw_arrays,
            "all_events": all_events,
            "run_numbers": run_numbers,
        }

    def process_record(
        self,
        record: str,
        raw_arrays: list,
        all_events: list,
        run_numbers: list,
        **kwargs,
    ):
        """
        Processes motor imagery epochs from all runs of a subject.

        For runs R03/R07/R11: T1 = left fist (class 0), T2 = right fist (class 1)
        For runs R04/R08/R12: T1 = both fists (class 2), T2 = both feet (class 3)

        Args:
            record (str): Subject identifier (e.g., "S001").
            raw_arrays (list): List of raw EEG arrays per run.
            all_events (list): List of event arrays per run.
            run_numbers (list): List of run numbers.

        Yields:
            dict: Preprocessed EEG segments with metadata.
        """
        subject_id = record
        write_pointer = 0

        for raw_data, events, run_num in zip(raw_arrays, all_events, run_numbers):
            if run_num in MI_RUNS_LEFT_RIGHT:
                label_map = {2: 0, 3: 1}  # T1=left fist, T2=right fist
            else:
                label_map = {2: 2, 3: 3}  # T1=both fists, T2=both feet

            for event in events:
                sample_start = event[0]
                event_id = event[2]

                if event_id == 1:
                    continue

                if event_id not in label_map:
                    continue

                sample_end = sample_start + EPOCH_SAMPLES
                if sample_end > raw_data.shape[1]:
                    continue

                trial_samples = raw_data[: self.num_channels, sample_start:sample_end]

                trial_meta = {
                    "subject_id": subject_id,
                    "run": run_num,
                    "event_id": event_id,
                    "motor_class": label_map[event_id],
                }

                write_pointer = yield from self._yield_windows(
                    trial_samples=trial_samples,
                    trial_meta=trial_meta,
                    write_ptr=write_pointer,
                    record_prefix=f"{record}_R{run_num:02d}",
                    start_at=0,
                    baseline_sample=None,
                )

    def set_records(
        self,
        root_path: str = "./physionet_mi_data/physionet.org/files/eegmmidb/1.0.0",
        **kwargs,
    ):
        """
        Returns the list of all subject directories in the dataset.

        Args:
            root_path (str): Root path of the EEGMMIDB dataset.

        Returns:
            List[str]: Sorted list of subject folder names.
        """
        assert os.path.exists(root_path), f"Dataset path not found: {root_path}"
        subjects = sorted(
            [d for d in os.listdir(root_path) if d.startswith("S") and d[1:].isdigit()]
        )
        return subjects


if __name__ == "__main__":
    label_transform = StackTransforms([Select("motor_class")])

    print("Starting PhysioNet MI preprocessing (4-class motor imagery)...")
    physionet_mi_dataset = PhysioNetMI(
        root_path="./physionet_mi_data/physionet.org/files/eegmmidb/1.0.0",
        trial_window_size=256,
        num_channels=64,
        stride=32,
        label_transform=label_transform,
        num_workers=8,
    )

    print(f"Total samples: {len(physionet_mi_dataset)}")

    os.makedirs("preprocessed_datasets", exist_ok=True)
    filename = "preprocessed_datasets/physionet_mi_multi_motor_imagery_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(physionet_mi_dataset, f)

    print(f"Saved to {filename}")
