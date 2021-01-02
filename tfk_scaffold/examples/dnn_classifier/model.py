from tfk_scaffold.common.model_wrapper import ModelWrapper
from tfk_scaffold.data.format import DataFormat, CsvFormatOptions, MNISTFormatOptions
from tfk_scaffold.registered import ModelRegistry

from typing import List, Tuple
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers


class Model(ModelWrapper):
    def __init__(self):
        super().__init__(
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.Adam(0.001),
            metrics=[metrics.Accuracy()],
        )

    @classmethod
    def model_name(cls) -> str:
        return 'dnn_classifier'
    
    @property
    def feature_names(self) -> List[str]:
        # TODO: Implementation
        return ['Image']
    
    @property
    def label_names(self) -> List[str]:
        # TODO: Implementation
        return ['Label']

    def construct_model(self, df: pd.DataFrame) -> keras.Model:
        unique_labels = df[self.label_names].unique()
        return keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(unique_labels)),
        ])

    @classmethod
    def fetch_data_impl(cls, path: str, fmt: DataFormat) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def preprocess(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['Image'] = df['Image'] / 255.0
            return df
        assert (fmt is DataFormat.MNIST)
        opts_train = fmt.create(kind='train')
        opts_test = fmt.create(kind='t10k')
        if opts_train is None or opts_test is None:
            raise RuntimeError("Error reading dataset")
    
        df_train = opts_train.read(path)
        df_test = opts_test.read(path)
        raw_df = pd.concat([df_train, df_test]).dropna()

        return raw_df, preprocess(raw_df.copy())

ModelRegistry.add(Model)
        
        