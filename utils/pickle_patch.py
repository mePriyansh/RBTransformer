import sys
import types
from dataset_classes import DEAP, DREAMER, SEED

from preprocessing.transformations import (
    DatasetReshape,
    subtract_by_one,
    add_by_one,
    StackTransforms,
    Select,
    Binarize,
    Tensorize,
    Normalize,
    BandDifferentialEntropy,
    SubtractBaseline,
    UnsqueezeDim,
    Lambda,
)


def patch_pickle_loading():
    dataset_main = types.ModuleType("__main__")

    dataset_main.DEAP = DEAP
    dataset_main.DREAMER = DREAMER
    dataset_main.SEED = SEED

    dataset_main.DatasetReshape = DatasetReshape
    dataset_main.subtract_by_one = subtract_by_one
    dataset_main.add_by_one = add_by_one
    dataset_main.StackTransforms = StackTransforms
    dataset_main.Select = Select
    dataset_main.Binarize = Binarize
    dataset_main.Tensorize = Tensorize
    dataset_main.Normalize = Normalize
    dataset_main.BandDifferentialEntropy = BandDifferentialEntropy
    dataset_main.SubtractBaseline = SubtractBaseline
    dataset_main.UnsqueezeDim = UnsqueezeDim
    dataset_main.Lambda = Lambda

    sys.modules["__main__"] = dataset_main
