from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Union, List, Literal, Callable, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class CsvFormatOptions:
    separator: str = ','
    delimiter: Optional[str] = None
    header: Optional[Union[int, List[int], Literal['infer']]] = 'infer'
    names: Optional[List[str]] = None
    index_col: Optional[Union[int, str, List[Union[int, str]],
                              Literal[False]]] = None
    usecols: Optional[Union[List[str], Callable[[str], bool]]] = None
    squeeze: bool = False
    prefix: Optional[str] = None
    mangle_dupe_cols: bool = True
    dtype: Optional[Union[str, Dict[str, np.dtype]]] = None
    engine: Optional[Union[Literal['c'], Literal['python']]] = None
    converters: Optional[Dict[str, Any]] = None
    true_values: Optional[List[Any]] = None
    false_values: Optional[List[Any]] = None
    skipininitialspace: bool = False
    skiprows: Optional[Union[List[Union[int, str]], Callable[[int, str], bool],
                             int, str]] = None
    skipfooter: int = 0
    nrows: Optional[int] = None
    infer_datetime_format: bool = False
    keep_date_col: bool = False
    date_parser: Optional[Any] = None
    dayfirst: bool = False
    cache_dates: bool = True
    iterator: bool = False
    chunksize: Optional[int] = None
    compression: Optional[Union[Literal['infer'], Literal['gzip'],
                                Literal['bz2'], Literal['zip'], Literal['xz'],
                                Literal['infer']]] = 'infer'
    thousands: Optional[str] = None
    decimal: str = "."
    lineterminator: Optional[str] = None
    quoatechar: Optional[str] = None
    quoting: int = 0
    doublequote: bool = True
    escapechar: Optional[str] = None
    comment: Optional[str] = None
    encodiing: Optional[str] = None
    dialect: Optional[str] = None
    error_bad_lines: bool = True
    warn_bad_lines: bool = True
    delim_whitespace: bool = False
    low_memory: bool = True
    memory_map: bool = False
    float_precision: Optional[str] = None
    storage_options: Optional[Dict[Any, Any]] = None

    def read(self, path):
        return pd.read_csv(path, **asdict(self))


@dataclass
class MNISTFormatOptions:
    kind: str

    def read(self, path):
        import gzip
        import numpy as np
        from urllib.parse import urlparse, urljoin

        img_bytes = bytes()
        lbl_bytes = bytes()

        def uri_validator(x):
            try:
                result = urlparse(x)
                return all([result.scheme, result.netloc, result.path])
            except:
                return False

        if uri_validator(path):
            import requests
            labels_path = urljoin(
                path, '{}-labels-idx1-ubyte.gz'.format(self.kind)
            )
            images_path = urljoin(
                path, '{}-images-idx3-ubyte.gz'.format(self.kind)
            )

            lbl_stream = requests.get(labels_path, stream=True)
            img_stream = requests.get(images_path, stream=True)

            lbl_bytes = lbl_stream.raw.read()
            img_bytes = img_stream.raw.read()

            lbl_bytes = gzip.decompress(lbl_bytes)
            img_bytes = gzip.decompress(img_bytes)
        else:
            import os.path
            labels_path = os.path.join(
                path, '{}-labels-idx1-ubyte.gz'.format(self.kind)
            )
            images_path = os.path.join(
                path, '{}-images-idx3-ubyte.gz'.format(self.kind)
            )

            with open(labels_path, 'rb') as f:
                lbl_bytes = f.read()
            with open(images_path, 'rb') as f:
                img_bytes = f.read()

        labels = np.frombuffer(lbl_bytes, dtype=np.uint8, offset=8)

        images = np.frombuffer(img_bytes, dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

        return pd.DataFrame([{
            "Image": np.array(img, dtype=np.float32),
            "Label": np.uint8(lbl)
        } for lbl, img in zip(labels, images)]).reset_index(drop=True)


class DataFormat(Enum):
    Csv = CsvFormatOptions
    MNIST = MNISTFormatOptions

    def create(self, *args,
               **kwargs) -> Union[CsvFormatOptions, MNISTFormatOptions, None]:
        if self is self.Csv:
            return CsvFormatOptions(*args, **kwargs)
        elif self is self.MNIST:
            return MNISTFormatOptions(*args, **kwargs)
        return None

    @staticmethod
    def from_str(s: str) -> Optional['DataFormat']:
        s = s.lower().strip()
        if s == 'csv':
            return DataFormat.Csv
        elif s == 'mnist':
            return DataFormat.MNIST
        else:
            return None