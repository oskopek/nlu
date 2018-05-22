from typing import TypeVar, Optional, Sequence, Dict

import numpy as np

K = TypeVar('K')
V = TypeVar('V')

UNK_TOKEN = '$unk$'
PAD_TOKEN = '$pad$'
BASE_VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1}


class MissingDict(Dict[K, V]):
    """Replace missing values with the default value, but do not insert them."""

    def __init__(self, *args, default_val: Optional[V] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_val: Optional[V] = default_val

    def __missing__(self, key: K) -> Optional[V]:
        return self.default_val


def map_all(sequences: Sequence[Sequence[K]], vocabulary: Dict[K, V]) -> Sequence[Sequence[V]]:
    return [[vocabulary[word] for word in seq] for seq in sequences]


def invert_dict(d: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in d.items()}


def invert_vocab(vocab: Dict[K, int]) -> Sequence[K]:
    assert sorted(vocab.values()) == np.arange(0, len(vocab))
    return [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
