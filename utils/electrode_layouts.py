"""
Standard electrode channel orderings for each EEG dataset and helpers to
resolve them to 3D scalp positions via MNE's standard_1005 montage.

Sources for the channel orderings:
    - DEAP: dataset documentation, Koelstra et al. 2012, fixed 32-channel layout
    - DREAMER: Emotiv EPOC 14-channel layout, Katsigiannis & Ramzan 2018
    - SEED: 62-channel extended 10-20, Zheng & Lu 2015
    - PhysioNet-MI: BCI2000 64-channel system, Schalk et al. 2004
"""

import numpy as np
import mne


DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

DREAMER_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

SEED_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]

PHYSIONET_MI_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]


DATASET_CHANNELS = {
    "deap": DEAP_CHANNELS,
    "dreamer": DREAMER_CHANNELS,
    "seed": SEED_CHANNELS,
    "physionet_mi": PHYSIONET_MI_CHANNELS,
}


# Aliases for dataset-specific channel names that aren't in MNE's
# standard_1005 montage. SEED's CB1/CB2 (China Brain) are inferior-occipital
# sites anatomically equivalent to OI1/OI2 in the extended 10-05 system.
_CHANNEL_ALIASES = {
    "CB1": "OI1",
    "CB2": "OI2",
}


def _build_position_lookup():
    """
    Builds a case-insensitive lookup from channel name -> 3D scalp position
    using MNE's standard_1005 montage (covers all standard 10-20/10-10/10-5
    positions including extended sites like T9, T10, Iz, etc.). Dataset-
    specific aliases (e.g., SEED's CB1/CB2) are resolved via _CHANNEL_ALIASES.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    lookup = {name.upper(): np.asarray(pos) for name, pos in positions.items()}
    for alias, target in _CHANNEL_ALIASES.items():
        if target.upper() in lookup:
            lookup[alias.upper()] = lookup[target.upper()]
    return lookup


_POSITION_LOOKUP = None


def get_position_lookup():
    global _POSITION_LOOKUP
    if _POSITION_LOOKUP is None:
        _POSITION_LOOKUP = _build_position_lookup()
    return _POSITION_LOOKUP


def get_electrode_positions(channel_names):
    """
    Resolves a list of channel names to a (N, 3) array of 3D scalp positions.

    Args:
        channel_names (list[str]): Names of channels in the order they appear
            in the dataset's data tensor.

    Returns:
        np.ndarray of shape (N, 3) with positions in meters.

    Raises:
        KeyError: if any channel name cannot be resolved against standard_1005.
    """
    lookup = get_position_lookup()
    positions = []
    missing = []
    for name in channel_names:
        key = name.upper()
        if key not in lookup:
            missing.append(name)
            continue
        positions.append(lookup[key])
    if missing:
        raise KeyError(
            f"Unknown channel names not in MNE standard_1005 montage: {missing}"
        )
    return np.stack(positions, axis=0)
