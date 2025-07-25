import os
import pickle
import scipy.io as scio
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
#                                    DREAMER-PREPROCESSING-CLASS                                    #
#####################################################################################################
class DREAMER(BaseDatasetPreprocessing):
    """Preprocessing Dataset class for DREAMER dataset for RBTransformer."""

    def __init__(
        self,
        root_path: str = "./DREAMER.mat",
        trial_window_size: int = 512,
        baseline_window_size: int = 128,
        num_channels: int = 14,
        num_baseline: int = 61,
        stride: int = 117,
        label_transform: Union[None, Callable] = None,
        num_workers: int = 0,
    ):
        super().__init__(
            root_path=root_path,
            num_channels=num_channels,
            num_baseline=num_baseline,
            stride=stride,
            trial_window_size=trial_window_size,
            baseline_window_size=baseline_window_size,
            label_transform=label_transform,
            num_workers=num_workers,
        )

    @staticmethod
    def read_record(record: str, root_path: str = "./DREAMER.mat", **kwargs) -> Dict:
        """
        Reads the DREAMER dataset from the .mat file and returns all subjects' EEG samples and labels.

        Args:
            record (str): Subject index (used later for selecting trials).
            root_path (str): Path to the DREAMER .mat file.

        Returns:
            Dict: Parsed .mat content containing all subjects' EEG samples and labels.
        """
        mat_data = scio.loadmat(root_path, verify_compressed_data_integrity=False)
        return {"mat_data": mat_data}

    def process_record(self, record: str, mat_data: Dict, **kwargs):
        """
        Processes EEG samples from a DREAMER record and yields fixed-length segments along with corresponding labels.

        Args:
            record (str): Record identifier (subject index).
            mat_data (Dict): Parsed .mat content from the DREAMER dataset.

        Yields:
            dict: For trial segments, returns {'eeg', 'key', 'info'}.
                For the baseline segment, returns {'eeg', 'key'}.
        """
        subject = record
        trial_len = len(
            mat_data["DREAMER"][0, 0]["Data"][0, 0]["EEG"][0, 0]["stimuli"][0, 0]
        )
        write_pointer = 0

        for trial_id in range(trial_len):
            trial_baseline_sample = mat_data["DREAMER"][0, 0]["Data"][0, subject][
                "EEG"
            ][0, 0]["baseline"][0, 0][trial_id, 0]
            trial_baseline_sample = trial_baseline_sample[
                :, : self.num_channels
            ].swapaxes(1, 0)
            trial_baseline_sample = (
                trial_baseline_sample[
                    :, : self.num_baseline * self.baseline_window_size
                ]
                .reshape(
                    self.num_channels, self.num_baseline, self.baseline_window_size
                )
                .mean(axis=1)
            )

            trial_meta_info = {
                "subject_id": subject,
                "trial_id": trial_id,
                "valence": mat_data["DREAMER"][0, 0]["Data"][0, subject][
                    "ScoreValence"
                ][0, 0][trial_id, 0],
                "arousal": mat_data["DREAMER"][0, 0]["Data"][0, subject][
                    "ScoreArousal"
                ][0, 0][trial_id, 0],
                "dominance": mat_data["DREAMER"][0, 0]["Data"][0, subject][
                    "ScoreDominance"
                ][0, 0][trial_id, 0],
            }

            trial_samples = mat_data["DREAMER"][0, 0]["Data"][0, subject]["EEG"][0, 0][
                "stimuli"
            ][0, 0][trial_id, 0]
            trial_samples = trial_samples[:, : self.num_channels].swapaxes(1, 0)

            write_pointer = yield from self._yield_windows(
                trial_samples=trial_samples,
                trial_meta=trial_meta_info,
                write_ptr=write_pointer,
                record_prefix=str(subject),
                start_at=0,
                baseline_sample=trial_baseline_sample,
            )

    def set_records(self, root_path: str = "./DREAMER.mat", **kwargs):
        """
        Returns the list of all records in the DREAMER dataset.

        Args:
            root_path (str): Root path of the DREAMER dataset.

        Returns:
            List[int]: List of subject indices.
        """
        assert os.path.exists(root_path)
        mat_data = scio.loadmat(root_path, verify_compressed_data_integrity=False)
        subject_len = len(mat_data["DREAMER"][0, 0]["Data"][0])
        return list(range(subject_len))


if __name__ == "__main__":
    #####################################################################################################
    #                            DREAMER-BINARY-VALENCE-DATASET-PREPROCESSING                           #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("valence"),
            Binarize(3.0),
        ]
    )

    dreamer_binary_valence_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_binary_valence_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_binary_valence_dataset, f)

    #####################################################################################################
    #                           DREAMER-BINARY-AROUSAL-DATASET-PREPROCESSING                            #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("arousal"),
            Binarize(3.0),
        ]
    )

    dreamer_binary_arousal_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_binary_arousal_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_binary_arousal_dataset, f)

    #####################################################################################################
    #                           DREAMER-BINARY-DOMINANCE-DATASET-PREPROCESSING                          #
    #####################################################################################################
    label_transform = StackTransforms(
        [
            Select("dominance"),
            Binarize(3.0),
        ]
    )

    dreamer_binary_dominance_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_binary_dominance_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_binary_dominance_dataset, f)

    #####################################################################################################
    #                            DREAMER-MULTI-VALENCE-DATASET-PREPROCESSING                            #
    #####################################################################################################
    label_transform = StackTransforms([Select("valence"), Lambda(subtract_by_one)])

    dreamer_multi_valence_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_multi_valence_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_multi_valence_dataset, f)

    #####################################################################################################
    #                            DREAMER-MULTI-AROUSAL-DATASET-PREPROCESSING                            #
    #####################################################################################################
    label_transform = StackTransforms([Select("arousal"), Lambda(subtract_by_one)])

    dreamer_multi_arousal_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_multi_arousal_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_multi_arousal_dataset, f)

    #####################################################################################################
    #                           DREAMER-MULTI-DOMINANCE-DATASET-PREPROCESSING                           #
    #####################################################################################################
    label_transform = StackTransforms([Select("dominance"), Lambda(subtract_by_one)])

    dreamer_multi_dominance_dataset = DREAMER(
        root_path="./DREAMER.mat",
        trial_window_size=512,
        baseline_window_size=128,
        num_channels=14,
        num_baseline=61,
        stride=117,
        label_transform=label_transform,
        num_workers=8,
    )

    filename = "preprocessed_datasets/dreamer_multi_dominance_dataset.pkl"
    with open(filename, "wb") as f:
        pickle.dump(dreamer_multi_dominance_dataset, f)
