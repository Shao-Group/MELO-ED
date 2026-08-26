from pathlib import Path
import random

import pandas as pd

from functions import data_reader1, data_reader3


def _input_file(path, argument_name):
    """Return an absolute input path and fail early with a useful message."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(
            f'{argument_name} does not exist or is not a file: {file_path}'
        )
    return file_path


def _max_edit_distance(file_path):
    """Find the largest edit distance in a three-column sequence-pair file."""
    max_ed = None
    with file_path.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(
                    f'{file_path}:{line_number}: expected at least three columns'
                )
            try:
                edit_distance = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f'{file_path}:{line_number}: invalid edit distance {fields[2]!r}'
                ) from exc
            max_ed = edit_distance if max_ed is None else max(max_ed, edit_distance)

    if max_ed is None:
        raise ValueError(f'test file is empty: {file_path}')
    return max_ed


def data_load_bd(rate, d1, d2, train_files, test_file, num_test,
                 num_train_valid, max_test_ed=15):
    """Load boundary-sampled train/validation data and a separate test set.

    ``train_files`` accepts any number of three-column sequence-pair files.
    Their names and sequence lengths are not interpreted by this loader.
    ``max_test_ed`` defaults to 15 to preserve the legacy loaders' behavior.
    Pass a different integer for a wider or narrower test edit-distance range,
    or None to infer the largest edit distance in ``test_file``.
    """
    if not 0 < rate < 1:
        raise ValueError('rate must be between 0 and 1')
    if d1 < 1 or d1 >= d2:
        raise ValueError('d1 must be at least 1 and less than d2')
    if num_test <= 0 or num_train_valid <= 0:
        raise ValueError('num_test and num_train_valid must be positive')

    if isinstance(train_files, (str, Path)):
        train_files = [train_files]
    train_files = [_input_file(path, 'train file') for path in train_files]
    if not train_files:
        raise ValueError('at least one train file is required')
    test_file = _input_file(test_file, 'test file')

    if max_test_ed is None:
        max_test_ed = _max_edit_distance(test_file)
    if max_test_ed < d2:
        raise ValueError('max_test_ed must be greater than or equal to d2')

    dataset_test, edit_distance_counts = data_reader1(
        max_test_ed, d1, d2, str(test_file), num_test
    )
    print(edit_distance_counts)
    df_test = pd.DataFrame(dataset_test)

    # data_reader3 selects edit distances in (start, end]. This reproduces the
    # legacy boundary sampling for any d1/d2: up to two positive EDs ending at
    # d1, and the two negative EDs d2 and d2 + 1.
    positive_start = max(0, d1 - 2)
    negative_start = d2 - 1
    negative_end = d2 + 1

    dataset_p = []
    dataset_n = []
    for train_file in train_files:
        dataset_p.extend(data_reader3(
            positive_start, d1, str(train_file), 0, num_train_valid, -1
        ))
        dataset_n.extend(data_reader3(
            negative_start, negative_end, str(train_file),
            0, num_train_valid, 1,
        ))

    datasets = dataset_p + dataset_n
    print(f'number_p:{len(dataset_p)}, number_n:{len(dataset_n)}')
    print('boundary sampling2')

    random.shuffle(datasets)
    train_range = int(len(datasets) * rate)
    df_tr = pd.DataFrame(datasets[:train_range])
    df_v = pd.DataFrame(datasets[train_range:])

    return df_tr, df_v, df_test
