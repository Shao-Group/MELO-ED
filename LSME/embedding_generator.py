import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from functions import leng_fea, mini_batch_cnn1
# Import model classes so full-model checkpoints saved by new_runner.py can be
# deserialized by torch.load.
from model_loss_train import *  # noqa: F403


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')


def _input_file(path):
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(
            f'input file does not exist or is not a file: {file_path}'
        )
    return file_path


def _validate_sequence(sequence, n_len, file_path, line_number):
    if len(sequence) != n_len:
        raise ValueError(
            f'{file_path}:{line_number}: expected a sequence of length '
            f'{n_len}, got {len(sequence)}'
        )


def read_sequence_pairs(data_file, n_len, num_pairs=10000):
    """Encode the first two columns of a paired-sequence input file."""
    data_file = _input_file(data_file)
    if n_len <= 0:
        raise ValueError('n_len must be positive')
    if num_pairs == 0 or num_pairs < -1:
        raise ValueError('num_pairs must be positive or -1 to read all pairs')

    seq_a = []
    seq_b = []
    with data_file.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    f'{data_file}:{line_number}: expected at least two columns'
                )
            _validate_sequence(fields[0], n_len, data_file, line_number)
            _validate_sequence(fields[1], n_len, data_file, line_number)
            seq_a.append(leng_fea(fields[0]))
            seq_b.append(leng_fea(fields[1]))
            if num_pairs != -1 and len(seq_a) == num_pairs:
                break

    if not seq_a:
        raise ValueError(f'no sequence pairs found in {data_file}')
    if num_pairs != -1 and len(seq_a) < num_pairs:
        raise ValueError(
            f'{data_file} contains only {len(seq_a)} non-empty pairs; '
            f'{num_pairs} requested'
        )
    return seq_a, seq_b


def read_sequences(data_file, n_len, num_sequences=-1):
    """Encode the first column of a single-sequence input file."""
    data_file = _input_file(data_file)
    if n_len <= 0:
        raise ValueError('n_len must be positive')
    if num_sequences == 0 or num_sequences < -1:
        raise ValueError(
            'num_sequences must be positive or -1 to read all sequences'
        )

    sequences = []
    with data_file.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            fields = line.split()
            if not fields:
                continue
            _validate_sequence(fields[0], n_len, data_file, line_number)
            sequences.append(leng_fea(fields[0]))
            if num_sequences != -1 and len(sequences) == num_sequences:
                break

    if not sequences:
        raise ValueError(f'no sequences found in {data_file}')
    if num_sequences != -1 and len(sequences) < num_sequences:
        raise ValueError(
            f'{data_file} contains only {len(sequences)} non-empty sequences; '
            f'{num_sequences} requested'
        )
    return sequences


def load_model(model_file):
    """Load a full Siamese model checkpoint while preserving its saved mode."""
    model_file = Path(model_file).expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(
            f'model file does not exist or is not a file: {model_file}'
        )
    model = torch.load(model_file, map_location=device, weights_only=False)
    model = model.to(device)
    print(f'Model loaded: {model_file}')
    return model


def _reshape_embeddings(output, num_b):
    """Infer m_dim and reshape a flat model output to [rows, num_b, m_dim]."""
    if num_b <= 0:
        raise ValueError('num_b must be positive')
    if output.ndim != 2:
        raise ValueError(
            f'expected a two-dimensional model output, got shape {output.shape}'
        )
    output_width = output.shape[1]
    if output_width % num_b != 0:
        raise ValueError(
            f'model output width {output_width} is not divisible by num_b={num_b}'
        )
    m_dim = output_width // num_b
    embeddings = (
        output.reshape(output.shape[0], num_b, m_dim)
        .detach().cpu().resolve_conj().resolve_neg().numpy()
    )
    return embeddings, m_dim


def generate_pair_embeddings(model, seq_a, seq_b, batch_size, num_b):
    """Generate embeddings for rows in a two-column paired input file."""
    if len(seq_a) != len(seq_b):
        raise ValueError('seq_a and seq_b must contain the same number of sequences')
    if not seq_a:
        raise ValueError('at least one sequence pair is required')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    embeddings_a = []
    embeddings_b = []
    inferred_m_dim = None
    num_batches = (len(seq_a) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_index in range(num_batches):
            input_a = mini_batch_cnn1(seq_a, batch_index, batch_size).to(device)
            input_b = mini_batch_cnn1(seq_b, batch_index, batch_size).to(device)
            output_a, output_b = model(input_a, input_b)
            embed_a, m_dim_a = _reshape_embeddings(output_a, num_b)
            embed_b, m_dim_b = _reshape_embeddings(output_b, num_b)
            if m_dim_a != m_dim_b:
                raise ValueError(
                    f'model produced inconsistent embedding dimensions: '
                    f'{m_dim_a} and {m_dim_b}'
                )
            if inferred_m_dim is not None and inferred_m_dim != m_dim_a:
                raise ValueError('model output width changed between batches')
            inferred_m_dim = m_dim_a
            embeddings_a.append(embed_a)
            embeddings_b.append(embed_b)

    return (
        np.concatenate(embeddings_a, axis=0),
        np.concatenate(embeddings_b, axis=0),
        inferred_m_dim,
    )


def generate_embeddings(model, sequences, batch_size, num_b):
    """Generate embeddings for one sequence collection using forward_once."""
    if not sequences:
        raise ValueError('at least one sequence is required')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not hasattr(model, 'forward_once'):
        raise TypeError('the loaded model does not define forward_once')

    embedding_batches = []
    inferred_m_dim = None
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_index in range(num_batches):
            inputs = mini_batch_cnn1(
                sequences, batch_index, batch_size
            ).to(device)
            output = model.forward_once(inputs)
            embeddings, m_dim = _reshape_embeddings(output, num_b)
            if inferred_m_dim is not None and inferred_m_dim != m_dim:
                raise ValueError('model output width changed between batches')
            inferred_m_dim = m_dim
            embedding_batches.append(embeddings)

    return np.concatenate(embedding_batches, axis=0), inferred_m_dim


def write_embeddings(output_file, datasets):
    """Write named embedding arrays to one HDF5 file."""
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not datasets:
        raise ValueError('at least one output dataset is required')
    if any(not name for name in datasets):
        raise ValueError('dataset names must not be empty')

    with h5py.File(output_file, 'w') as output_handle:
        for name, embeddings in datasets.items():
            output_handle.create_dataset(name, data=embeddings)
    print(f'Embeddings created: {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings from a trained Siamese model.',
    )
    parser.add_argument('--n_len', '--N_len', dest='n_len', type=int,
                        required=True, help='Expected sequence length')
    parser.add_argument('--model_file', type=str, required=True,
                        help='Full Siamese model checkpoint saved by new_runner.py')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output HDF5 file')
    parser.add_argument('--num_b', type=int, required=True,
                        help='Number of embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Inference batch size')
    parser.add_argument(
        '--num_records', '--num_pairs', dest='num_records', type=int,
        default=None,
        help='Records to read per input file; -1 reads all. The paired-file '
             'default is 10000 for legacy compatibility; one-column modes '
             'default to all records.',
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--data_file', type=str,
        help='Legacy paired input: first two columns contain paired sequences',
    )
    input_group.add_argument(
        '--input_files', type=str, nargs=2, metavar=('FILE_A', 'FILE_B'),
        help='Two independent one-column sequence files',
    )
    input_group.add_argument(
        '--input_file', type=str,
        help='One single-column sequence file',
    )
    parser.add_argument(
        '--dataset_names', type=str, nargs='+', default=None, metavar='NAME',
        help='HDF5 dataset name(s). Defaults depend on the input mode.',
    )
    args = parser.parse_args()

    if args.n_len <= 0:
        parser.error('--n_len must be positive')
    if args.num_b <= 0:
        parser.error('--num_b must be positive')
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if args.num_records is not None and (
        args.num_records == 0 or args.num_records < -1
    ):
        parser.error('--num_records must be positive or -1')

    model = load_model(args.model_file)

    if args.data_file is not None:
        dataset_names = args.dataset_names or ['embed_a', 'embed_b']
        if len(dataset_names) != 2:
            parser.error('paired input requires exactly two --dataset_names')
        if len(set(dataset_names)) != 2:
            parser.error('--dataset_names must be unique')
        num_pairs = 10000 if args.num_records is None else args.num_records
        seq_a, seq_b = read_sequence_pairs(
            args.data_file, args.n_len, num_pairs
        )
        embed_a, embed_b, m_dim = generate_pair_embeddings(
            model, seq_a, seq_b, args.batch_size, args.num_b
        )
        datasets = {
            dataset_names[0]: embed_a,
            dataset_names[1]: embed_b,
        }
        counts = [len(seq_a), len(seq_b)]

    elif args.input_files is not None:
        dataset_names = args.dataset_names or ['embed_a', 'embed_d']
        if len(dataset_names) != 2:
            parser.error('two-file input requires exactly two --dataset_names')
        if len(set(dataset_names)) != 2:
            parser.error('--dataset_names must be unique')
        num_sequences = -1 if args.num_records is None else args.num_records
        sequences_a = read_sequences(
            args.input_files[0], args.n_len, num_sequences
        )
        sequences_b = read_sequences(
            args.input_files[1], args.n_len, num_sequences
        )
        embed_a, m_dim_a = generate_embeddings(
            model, sequences_a, args.batch_size, args.num_b
        )
        embed_b, m_dim_b = generate_embeddings(
            model, sequences_b, args.batch_size, args.num_b
        )
        if m_dim_a != m_dim_b:
            raise ValueError(
                f'model produced inconsistent embedding dimensions: '
                f'{m_dim_a} and {m_dim_b}'
            )
        m_dim = m_dim_a
        datasets = {
            dataset_names[0]: embed_a,
            dataset_names[1]: embed_b,
        }
        counts = [len(sequences_a), len(sequences_b)]

    else:
        dataset_names = args.dataset_names or ['embed_a']
        if len(dataset_names) != 1:
            parser.error('single-file input requires exactly one --dataset_names')
        num_sequences = -1 if args.num_records is None else args.num_records
        sequences = read_sequences(
            args.input_file, args.n_len, num_sequences
        )
        embeddings, m_dim = generate_embeddings(
            model, sequences, args.batch_size, args.num_b
        )
        datasets = {dataset_names[0]: embeddings}
        counts = [len(sequences)]

    print(
        f'Embedding shape: num_b={args.num_b}, inferred m_dim={m_dim}; '
        f'input counts={counts}'
    )
    write_embeddings(args.output_file, datasets)


if __name__ == '__main__':
    main()
