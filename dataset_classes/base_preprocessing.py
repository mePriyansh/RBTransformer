import numpy as np
from collections import defaultdict
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from preprocessing.transformations import (
    Tensorize,
    Normalize,
    BandDifferentialEntropy,
    SubtractBaseline,
    StackTransforms,
    UnsqueezeDim,
)
from typing import Any, Callable, List, Tuple, Dict


#####################################################################################################
#                                      BASE DATASET-PREPROCESSING-CLASS                             #
#####################################################################################################
class BaseDatasetPreprocessing(Dataset):
    """
    BaseDatasetPreprocessing class for dataset preprocessing in RBTransformer.
    Subclassed by SEED, DEAP, and DREAMER.
    """

    def __init__(self, **kwargs):
        self.trial_window_size = kwargs.get("trial_window_size")
        self.baseline_window_size = kwargs.get("baseline_window_size")
        self.num_channels = kwargs.get("num_channels")
        self.num_baseline = kwargs.get("num_baseline")
        self.stride = kwargs.get("stride")
        self.label_transform = kwargs.get("label_transform")
        self.num_workers = kwargs.get("num_workers")
        self.tensorize = Tensorize()
        self._eeg_memory: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._info_memory: List[Dict] = []
        self.dataset_name = self.__class__.__name__.replace("Dataset", "")
        self.apply_to_baseline = (
            self.num_baseline is not None and self.baseline_window_size is not None
        )
        self.preprocessing_transformations = StackTransforms(
            [
                Normalize(apply_to_baseline=self.apply_to_baseline),
                BandDifferentialEntropy(apply_to_baseline=self.apply_to_baseline),
                SubtractBaseline(),
                UnsqueezeDim(),
            ]
        )

        records = self.set_records(**kwargs)

        if self.num_workers == 0:
            # Sequential preprocessing of records
            pbar = tqdm(
                total=len(records),
                disable=False,
                desc=f"Preprocessing {self.dataset_name} Dataset",
                position=0,
                leave=None,
            )
            for record_id, record in enumerate(records):
                infos, eegs = self.handle_record(
                    record=record,
                    record_id=record_id,
                    read_record=self.read_record,
                    process_record=self.process_record,
                    **kwargs,
                )
                self._info_memory.extend(infos)
                for tag, data in eegs.items():
                    self._eeg_memory[tag].update(data)
                pbar.update(1)
            pbar.close()
        else:
            # Parallel preprocessing of records
            with tqdm_joblib(
                total=len(records), desc=f"Preprocessing {self.dataset_name} Dataset"
            ) as _:
                results = Parallel(n_jobs=self.num_workers)(
                    delayed(self.handle_record)(
                        record=record,
                        record_id=record_id,
                        read_record=self.read_record,
                        process_record=self.process_record,
                        **kwargs,
                    )
                    for record_id, record in enumerate(records)
                )
            for infos, eegs in results:
                self._info_memory.extend(infos)
                for tag, data in eegs.items():
                    self._eeg_memory[tag].update(data)

    def handle_record(
        self,
        record: Any,
        record_id: int,
        read_record: Callable,
        process_record: Callable,
        **kwargs,
    ) -> Tuple[List[Dict], Dict[str, Dict[str, Any]]]:
        """
        Reads and preprocesses an EEG record into fixed-length segments.

        Args:
            record (Any): Record identifier.
            record_id (int): Index of the record.
            read_record (Callable): Read record function.
            process_record (Callable): Processing record function.
            **kwargs: Additional arguments passed to both functions.

        Returns:
            Tuple:
                - List[Dict]: Metadata for each EEG segment.
                - Dict[str, Dict[str, Any]]: EEG segments grouped by record tag.
        """
        kwargs = dict(kwargs)
        kwargs["record"] = record
        kwargs.update(read_record(**kwargs))

        gen = process_record(**kwargs)
        record_tag = f"_record_{record_id}"
        infos: List[Dict] = []
        eegs: Dict[str, Dict[str, Any]] = {record_tag: {}}

        for obj in gen:
            if obj and "eeg" in obj and "key" in obj:
                eegs[record_tag][obj["key"]] = obj["eeg"]
            if obj and "info" in obj:
                obj["info"]["_record_id"] = record_tag
                infos.append(obj["info"])

        return infos, eegs

    def _yield_windows(
        self,
        trial_samples: np.ndarray,
        trial_meta: dict,
        write_ptr: int,
        record_prefix: str,
        start_at: int = 0,
        baseline_sample: np.ndarray | None = None,
    ):
        """
        Chunks trial and baseline EEG records into fixed-length segments.

        Args:
            trial_samples (np.ndarray): Raw EEG trial data.
            trial_meta (dict): Metadata for the trial.
            write_ptr (int): Global index for segment IDs.
            record_prefix (str): Prefix for segment keys.
            start_at (int): Start index for segmenting.
            baseline_sample (np.ndarray | None): Optional baseline segment.

        Yields:
            dict: For trial segments, returns {'eeg', 'key', 'info'}.
                For the baseline segment, returns {'eeg', 'key'}.
        """

        window_size = self.trial_window_size

        if window_size <= 0:
            window_size = trial_samples.shape[1] - start_at

        end_at = start_at + window_size
        baseline_done = baseline_sample is None

        while end_at <= trial_samples.shape[1]:
            clip = trial_samples[:, start_at:end_at]

            if baseline_sample is not None:
                transformed = self.preprocessing_transformations(
                    eeg=clip, baseline=baseline_sample
                )
                t_eeg = transformed["eeg"]
                t_baseline = transformed["baseline"]
            else:
                t_eeg = self.preprocessing_transformations(eeg=clip)["eeg"]

            if not baseline_done:
                baseline_id = f"{record_prefix}_{write_ptr}"
                yield {"eeg": t_baseline, "key": baseline_id}
                trial_meta["baseline_id"] = baseline_id
                write_ptr += 1
                baseline_done = True

            clip_id = f"{record_prefix}_{write_ptr}"
            info = {"start_at": start_at, "end_at": end_at, "clip_id": clip_id}
            info.update(trial_meta)
            yield {"eeg": t_eeg, "key": clip_id, "info": info}
            write_ptr += 1

            start_at += self.stride
            end_at = start_at + window_size

        return write_ptr

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of preprocessed EEG windows in the dataset.
        """
        return len(self._info_memory)

    def __getitem__(self, index: int) -> Tuple:
        """
        Retrieves a preprocessed EEG sample and the corresponding label.

        Args:
            index (int): Index of the preprocessed EEG sample to retrieve.

        Returns:
            Tuple:
                - signal (torch.Tensor): Preprocessed EEG sample.
                - label (Any): Label of corresponding EEG sample.
        """
        info = self._info_memory[index]
        eeg_index = str(info["clip_id"])
        eeg_record = str(info["_record_id"])
        eeg = self._eeg_memory[eeg_record][eeg_index]

        label = info

        if (
            self.num_baseline is not None
            and self.baseline_window_size is not None
            and "baseline_id" in info
        ):
            baseline_index = str(info["baseline_id"])
            baseline = self._eeg_memory[eeg_record][baseline_index]
            signal = self.tensorize(eeg=eeg, baseline=baseline)["eeg"]
        else:
            signal = self.tensorize(eeg=eeg)["eeg"]

        if self.label_transform:
            label = self.label_transform(y=info)["y"]

        return signal, label
