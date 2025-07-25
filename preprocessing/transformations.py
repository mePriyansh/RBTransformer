import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter
from typing import Callable, Dict, Union, List, Tuple


#####################################################################################################
#                                      BaseTransform Class                                          #
#####################################################################################################
class BaseTransform:
    """
    Base class for all EEG data transformations.
    """

    def __init__(self):
        self._additional_targets: Dict[str, str] = {}

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        if args:
            raise KeyError("Please pass data as named parameters.")
        res = {}

        params = self.get_params()

        if self.targets_as_params:
            assert all(key in kwargs for key in self.targets_as_params), (
                "{} requires {}".format(self.__class__.__name__, self.targets_as_params)
            )
            targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
            params_dependent_on_targets = self.get_params_dependent_on_targets(
                targets_as_params
            )
            params.update(params_dependent_on_targets)

        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                res[key] = target_function(arg, **params)
        return res

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params(self) -> Dict:
        return {}

    def get_params_dependent_on_targets(self, params: Dict[str, any]) -> Dict[str, any]:
        return {}

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def add_targets(self, additional_targets: Dict[str, str]):
        self._additional_targets = additional_targets

    @property
    def targets(self) -> Dict[str, Callable]:
        raise NotImplementedError(
            "Method targets is not implemented in class " + self.__class__.__name__
        )

    def apply(self, *args, **kwargs) -> any:
        raise NotImplementedError(
            "Method apply is not implemented in class " + self.__class__.__name__
        )

    @property
    def repr_body(self) -> Dict:
        return {}

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for i, (k, v) in enumerate(self.repr_body.items()):
            if i:
                format_string += ", "
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ")"
        return format_string


#####################################################################################################
#                                       EEGTransform Class                                          #
#####################################################################################################
class EEGTransform(BaseTransform):
    """
    Base class for EEG-specific transformations.
    """

    def __init__(self, apply_to_baseline: bool = False):
        super(EEGTransform, self).__init__()
        self.apply_to_baseline = apply_to_baseline
        if apply_to_baseline:
            self.add_targets({"baseline": "eeg"})

    @property
    def targets(self):
        return {"eeg": self.apply}

    def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
        raise NotImplementedError(
            "Method apply is not implemented in class " + self.__class__.__name__
        )

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{"apply_to_baseline": self.apply_to_baseline})


#####################################################################################################
#                                     LabelTransform Class                                          #
#####################################################################################################
class LabelTransform(BaseTransform):
    """
    Base class for label-specific transformations.
    """

    @property
    def targets(self):
        return {"y": self.apply}

    def apply(self, y: any, **kwargs) -> any:
        raise NotImplementedError(
            "Method apply is not implemented in class " + self.__class__.__name__
        )


#####################################################################################################
#                                BandDifferentialEntropy Class                                      #
#####################################################################################################
class BandDifferentialEntropy(EEGTransform):
    """
    Computes the Band Differential Entropy (BDE) for EEG signals across predefined frequency bands.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        order: int = 5,
        band_dict: Dict[str, Tuple[int, int]] = {
            "theta": [4, 8],
            "alpha": [8, 14],
            "beta": [14, 31],
            "gamma": [31, 49],
        },
        apply_to_baseline: bool = False,
    ):
        super().__init__(apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

    def __call__(
        self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = self._butter_bandpass(
                    low, high, fs=self.sampling_rate, order=self.order
                )
                c_list.append(self._calculate_differential_entropy(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    def _calculate_differential_entropy(self, eeg: np.ndarray) -> np.ndarray:
        return 0.5 * np.log2(2 * np.pi * np.e * np.var(eeg))

    def _butter_bandpass(self, low_cut, high_cut, fs, order=5):
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body,
            **{
                "sampling_rate": self.sampling_rate,
                "order": self.order,
                "band_dict": {...},
            },
        )


#####################################################################################################
#                                    SubtractBaseline Class                                         #
#####################################################################################################
class SubtractBaseline(EEGTransform):
    """
    Subtracts the baseline signal from corresponding trial EEG samples.
    """

    def __init__(self):
        super(SubtractBaseline, self).__init__(apply_to_baseline=False)

    def __call__(
        self, *args, eeg: any, baseline: Union[any, None] = None, **kwargs
    ) -> Dict[str, any]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: any, **kwargs) -> any:
        if kwargs["baseline"] is None:
            return eeg

        assert kwargs["baseline"].shape == eeg.shape, (
            f"The shape of baseline signals ({kwargs['baseline'].shape}) must match the input signal ({eeg.shape})."
        )
        return eeg - kwargs["baseline"]

    @property
    def targets_as_params(self):
        return ["baseline"]

    def get_params_dependent_on_targets(self, params):
        return {"baseline": params["baseline"]}


#####################################################################################################
#                                     StackTransforms Class                                         #
#####################################################################################################
class StackTransforms(BaseTransform):
    """
    Chains multiple preprocessing transforms and applies them sequentially.
    """

    def __init__(self, transforms: List[Callable]):
        super(StackTransforms, self).__init__()
        self.transforms = transforms

    def __call__(self, *args, **kwargs) -> any:
        if args:
            raise KeyError("Please pass data as named parameters.")

        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ","
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


#####################################################################################################
#                                        Lambda Class                                               #
#####################################################################################################
class Lambda(BaseTransform):
    """
    Applies a custom transformation function to specified targets (EEG, baseline, or labels).
    """

    def __init__(
        self, lambda_fun: Callable, targets: List[str] = ["eeg", "baseline", "y"]
    ):
        super(Lambda, self).__init__()
        self._targets = targets
        self.lambda_fun = lambda_fun

    @property
    def targets(self) -> Dict[str, Callable]:
        return {target: self.apply for target in self._targets}

    def apply(self, *args, **kwargs) -> any:
        return self.lambda_fun(args[0])

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        return super().__call__(*args, **kwargs)

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{"lambda_fun": self.lambda_fun, "targets": [...]}
        )


#####################################################################################################
#                                       Normalize Class                                             #
#####################################################################################################
class Normalize(EEGTransform):
    """
    Applies Mean-Standard Deviation Normalization on EEG samples.
    """

    def __init__(
        self,
        mean: Union[np.ndarray, None] = None,
        std: Union[np.ndarray, None] = None,
        axis: Union[int, None] = None,
        apply_to_baseline: bool = False,
    ):
        super(Normalize, self).__init__(apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std
        self.axis = axis

    def __call__(
        self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs):
        if (self.mean is None) or (self.std is None):
            if self.axis is None:
                mean = eeg.mean()
                std = eeg.std()
            else:
                mean = eeg.mean(axis=self.axis, keepdims=True)
                std = eeg.std(axis=self.axis, keepdims=True)
        else:
            if self.axis is None:
                axis = 1
            else:
                axis = self.axis
            assert len(self.mean) == eeg.shape[axis], (
                f"The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given mean's dimension {len(self.mean)}."
            )
            assert len(self.std) == eeg.shape[axis], (
                f"The given normalized axis has {eeg.shape[axis]} dimensions, which does not match the given std's dimension {len(self.std)}."
            )
            shape = [1] * len(eeg.shape)
            shape[axis] = -1
            mean = self.mean.reshape(*shape)
            std = self.std.reshape(*shape)

        if np.any(std == 0):
            std[std == 0] = 1

        return (eeg - mean) / std

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{"mean": self.mean, "std": self.std, "axis": self.axis}
        )


#####################################################################################################
#                                       Select Class                                                #
#####################################################################################################
class Select(LabelTransform):
    """
    Extracts specified emotion dimension(s)—Valence, Arousal, or Dominance—from a label dictionary.
    """

    def __init__(self, key: Union[str, List]):
        super(Select, self).__init__()
        self.key = key
        self.select_list = isinstance(key, list) or isinstance(key, tuple)

    def __call__(self, *args, y: Dict, **kwargs) -> Union[int, float, List]:
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Dict, **kwargs) -> Union[int, float, List]:
        assert isinstance(y, dict), (
            f"The transform Select only accepts label dict as input, but obtained {type(y)} instead."
        )
        if self.select_list:
            return [y[k] for k in self.key]
        return y[self.key]

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{"key": self.key})


#####################################################################################################
#                                       Binarize Class                                              #
#####################################################################################################
class Binarize(LabelTransform):
    """
    Converts continuous emotion scores into high and low classes for a given threshold.
    """

    def __init__(self, threshold: float):
        super(Binarize, self).__init__()
        self.threshold = threshold

    def __call__(self, *args, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Union[int, float, List], **kwargs) -> Union[int, List]:
        if isinstance(y, list):
            return [self.binarize(l) for l in y]
        return self.binarize(y)

    def binarize(self, y: Union[int, float]) -> int:
        if np.isnan(y):
            return np.random.randint(2)
        return int(y >= self.threshold)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{"threshold": self.threshold})


#####################################################################################################
#                                   UnsqueezeDim Class                                              #
#####################################################################################################
class UnsqueezeDim(EEGTransform):
    """
    Adds a leading dimension to EEG samples, converting [channels, bands] to [1, channels, bands] for batching.
    """

    def __call__(
        self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return eeg[np.newaxis, ...]


#####################################################################################################
#                                       Tensorize Class                                             #
#####################################################################################################
class Tensorize(EEGTransform):
    """
    Converts EEG data from NumPy arrays to PyTorch tensors.
    """

    def __init__(self, apply_to_baseline: bool = False):
        super(Tensorize, self).__init__(apply_to_baseline=apply_to_baseline)

    def __call__(
        self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        return torch.from_numpy(eeg).float()


#####################################################################################################
#                                   DatasetReshape Class                                            #
#####################################################################################################
class DatasetReshape(Dataset):
    """
    Reshapes flattened BDE features into [batch, num_electrodes, bands] format for model input.
    """

    def __init__(self, X, y, num_electrodes=14):
        self.X = torch.tensor(
            X.reshape(-1, num_electrodes, 4), dtype=torch.float32
        ).squeeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#####################################################################################################
#                                     Utility Lamdba Functions                                      #
#####################################################################################################


def subtract_by_one(x):
    return int(x) - 1


def add_by_one(x):
    return int(x) + 1
