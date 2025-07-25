def get_num_electrodes(dataset):
    """
    Returns the number of EEG electrodes for a given dataset.
    """
    return {
        "seed": 62,
        "deap": 32,
        "dreamer": 14,
    }[dataset]


def get_num_classes(dataset, task_type):
    """
    Returns the number of output classes based on dataset and classification type.
    """
    if dataset == "seed":
        return 3
    elif dataset == "deap":
        return 2 if task_type == "binary" else 9
    elif dataset == "dreamer":
        return 2 if task_type == "binary" else 5
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
