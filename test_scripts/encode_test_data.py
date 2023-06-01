"""Simple throwaway script for encoding the test data."""
import os
import random
import numpy as np
from .test_constants import AA_DICT


def encode_seq_data(start_dir):
    """Encodes the test data under the test data folder
    in the project directory (start_dir).

    Raises:
        ValueError: A ValueError is raised if the expected test
            data is not found.
    """
    os.chdir(os.path.join(start_dir, "test_data"))
    if "converted_seqs.txt" not in os.listdir():
        raise ValueError("Expected test data file not found!")
    if "encoded_test_data.npy" in os.listdir():
        os.chdir(start_dir)
        return

    encoded_seqs = []
    with open("converted_seqs.txt", "r") as fhandle:
        for line in fhandle:
            seq = line.strip()
            encoded_seqs.append(encode_seq(seq))
    encoded_seqs = np.stack(encoded_seqs)
    np.save("encoded_test_data.npy", encoded_seqs)

    mutagenized_data = mutagenize_parent_seq(encoded_seqs[0,:])
    np.save("decoys.npy", mutagenized_data)
    os.chdir(start_dir)


def encode_seq(seq):
    """Encodes a single sequence as a uint8 array."""
    encoding = [AA_DICT[a] for a in seq]
    return np.array(encoding, dtype = np.uint8)


def mutagenize_parent_seq(parent_seq, num_decoys = 50):
    """Generates heavily mutated versions of the parent
    sequence that will illustrate the model's ability
    to recognize sequences distinct from its training
    set."""
    random.seed(123)
    decoys = []
    for i in range(num_decoys):
        decoy = parent_seq.copy()
        for i in range(parent_seq.shape[0]):
            if random.randint(0,9) == 9:
                decoy[i] = random.randint(0,20)
        decoys.append(np.array(decoy, dtype=np.uint8))

    return np.stack(decoys)
