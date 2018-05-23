from typing import List, Optional, Callable

import numpy as np
import pandas as pd

from sct.data.preprocessing import Preprocessing
from sct.data.utils import create_sentence_indexer


def preprocessing_compare(preprocessing: Preprocessing,
                          inp: List[str],
                          exp: List[str],
                          call_on_result: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> None:
    sentence_indexer = create_sentence_indexer(n_sentences=1, n_endings=0)
    preprocessing.sentence_indexer = sentence_indexer
    df = pd.DataFrame({"sentence1": inp})
    df_target = pd.DataFrame({"sentence1": exp})

    df_processed = preprocessing.transform(df, evaluate=False)
    if call_on_result is not None:
        df_processed = call_on_result(df_processed)
    assert np.array_equal(df_target.values, df_processed.values)


def test_default_flags_is_identity() -> None:
    preprocessing_compare(
            Preprocessing(), inp=["Lol, lol2!!!", "Test....", "Test ."], exp=["Lol, lol2!!!", "Test....", "Test ."])


def test_standardize() -> None:
    preprocessing_compare(
            Preprocessing(standardize=True),
            inp=["Lol, lol2!!!", "Test....", "Test ."],
            exp=["lol, lol2!!!", "test....", "test ."])


def test_vocab_downsize() -> None:
    # data has 6 words:
    # "word3": 1, "word2": 1, ",": 1, "word1": 2, "test": 2, ".": 3
    preprocessing = Preprocessing()

    def downsize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsize data to three most common words (+ $unk$ and $pad$)
        """
        vocabs = preprocessing.vocab(df, vocab_downsize=2 + 3)
        return preprocessing.vocab(df, vocab_downsize=vocabs)

    preprocessing_compare(
            preprocessing,
            inp=["word1 , word1 word2", "test .", "test . word3 ."],
            exp=["word1 $unk$ word1 $unk$", "test .", "test . $unk$ ."],
            call_on_result=downsize)
