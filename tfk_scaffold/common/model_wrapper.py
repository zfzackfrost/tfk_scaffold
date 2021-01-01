import os.path
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable, Any, cast

from tensorflow import keras
import pandas as pd

KerasLoss = Union[keras.losses.Loss, Callable[[Any, Any], Any], str]
KerasOptimizer = Union[keras.optimizers.Optimizer, str]
KerasMetric = Union[keras.metrics.Metric, str]
class ModelWrapper(ABC):

    def __init__(self, loss: KerasLoss, optimizer: KerasOptimizer, metrics: List[KerasMetric] = []):
        self.__model: Optional[keras.Model] = None
        self.__loss = loss
        self.__optimizer = optimizer
        self.__metrics = metrics
    
    @property
    def model(self) -> Optional[keras.Model]:
        return self.__model
    
    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def label_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def construct_model(self, df: pd.DataFrame) -> keras.Model:
        raise NotImplementedError()
    
    def save(self, path):
        if self.model is not None:
            if not os.path.isabs(path):
                scriptdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
                models_cache = os.path.join(scriptdir, '.models')
                path = os.path.join(models_cache, path)
            keras.models.save_model(self.model, path, overwrite=True)

    def load(self, path):
        if not os.path.isabs(path):
            scriptdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
            models_cache = os.path.join(scriptdir, '.models')
            path = os.path.join(models_cache, path)
        if not os.path.exists(path):
            return
        self.__model = cast(Any, keras.models.load_model(path))

    def fit(self, df: pd.DataFrame):
        self.__ensure_model(df)

        self.save(self.model_name())

    def evaluate(self, df: pd.DataFrame):
        self.load(self.model_name())
        self.__ensure_model(df)

    def predict(self, df: pd.DataFrame):
        self.load(self.model_name())
        self.__ensure_model(df)

    def __ensure_model(self, df: pd.DataFrame):
        if self.model is None:
            m = self.construct_model(df)
            m.compile(loss=self.__loss, optimizer=self.__optimizer, metrics=self.__metrics)
            self.__model = m

