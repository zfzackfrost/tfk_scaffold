import pandas as pd

from dataclasses import dataclass

import numpy as np
from typing import Union, Literal, Optional

import os
import pickle

@dataclass
class DataSet:
    train: pd.DataFrame = pd.DataFrame()
    test: pd.DataFrame = pd.DataFrame()
    raw: pd.DataFrame = pd.DataFrame()

    def log(
        self,
        elem: Union[Literal['train'], Literal['test'], Literal['raw']],
        tail: Optional[int] = None,
        head: Optional[int] = None
    ):
        assert elem in ['train', 'test', 'raw'], "Invalid value for elem"
        assert tail is not None or head is not None, "A value other than None is required for `tail` or `head`"

        df = getattr(self, elem)
        df = df.head(head) if head is not None else df.tail(tail)
        with np.printoptions(precision=5, floatmode='fixed'):
            print(df)

    def save(
        self,
        path,
        compression: Union[Literal['infer', Literal['gzip'], Literal['bz2'],
                                   Literal['zip'], Literal['xz'],
                                   None]] = "infer"
    ):
        packed_df = pd.DataFrame({
            "train": [self.train],
            "test": [self.test],
            "raw": [self.raw],
        })
        packed_df.to_pickle(path, compression=compression, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self,
        path,
        compression: Union[Literal['infer', Literal['gzip'], Literal['bz2'],
                                   Literal['zip'], Literal['xz'],
                                   None]] = "infer"
    ):
        packed_df =  pd.read_pickle(path, compression=compression)
        self.train = packed_df['train'][0]
        self.test = packed_df['test'][0]
        self.raw = packed_df['raw'][0]
