from typing import Dict, Type, Optional

from tfk_scaffold import ModelWrapper

class ModelRegistry:
    __models: Dict[str, Type[ModelWrapper]] = {}
    
    @classmethod
    def add(cls, model_type: Type[ModelWrapper]):
        k = model_type.model_name()
        cls.__models[k] = model_type

    @classmethod
    def has_model(cls, model_name: str):
        return model_name in cls.__models

    @classmethod
    def get(cls, model_name: str) -> Optional[Type[ModelWrapper]]:
        if cls.has_model(model_name):
            return cls.__models[model_name]
        return None