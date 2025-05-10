from typing import Optional, Dict, Union, Any, List
import random

from bitmind.types import ModelConfig, ModelTask


class ModelRegistry:
    """
    Registry for managing generative models.
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}

    def register(self, model_config: ModelConfig) -> None:
        self.models[model_config.path] = model_config

    def register_all(self, model_configs: List[ModelConfig]) -> None:
        for config in model_configs:
            self.register(config)

    def get_model(self, path: str) -> Optional[ModelConfig]:
        return self.models.get(path)

    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.models.copy()

    def get_models_by_task(self, task: ModelTask) -> Dict[str, ModelConfig]:
        return {
            path: config for path, config in self.models.items() if config.task == task
        }

    def get_model_names_by_task(self, task: ModelTask) -> Dict[str, ModelConfig]:
        return [path for path, config in self.models.items() if config.task == task]

    def get_models_by_tag(self, tag: str) -> Dict[str, ModelConfig]:
        return {
            path: config for path, config in self.models.items() if tag in config.tags
        }

    def get_model_names_by_task(self, task: ModelTask) -> List[str]:
        return list(self.get_models_by_task(task).keys())

    @property
    def t2i_models(self) -> Dict[str, ModelConfig]:
        return self.get_models_by_task(ModelTask.TEXT_TO_IMAGE)

    @property
    def t2v_models(self) -> Dict[str, ModelConfig]:
        return self.get_models_by_task(ModelTask.TEXT_TO_VIDEO)

    @property
    def i2i_models(self) -> Dict[str, ModelConfig]:
        return self.get_models_by_task(ModelTask.IMAGE_TO_IMAGE)

    @property
    def i2v_models(self) -> List[str]:
        return self.get_models_by_task(ModelTask.IMAGE_TO_VIDEO)

    @property
    def t2i_model_names(self) -> List[str]:
        return list(self.t2i_models.keys())

    @property
    def t2v_model_names(self) -> List[str]:
        return list(self.t2v_models.keys())

    @property
    def i2i_model_names(self) -> List[str]:
        return list(self.i2i_models.keys())

    @property
    def i2v_model_names(self) -> List[str]:
        return list(self.i2v_models.keys())

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def select_random_model(self, task: Optional[Union[ModelTask, str]] = None) -> str:
        if isinstance(task, str):
            task = ModelTask(task.lower())

        if task is None:
            task = random.choice(list(ModelTask))

        model_names = self.get_model_names_by_task(task)
        if not model_names:
            raise ValueError(f"No models available for task: {task}")

        return random.choice(model_names)

    def get_model_dict(self, model_name: str) -> Dict[str, Any]:
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        return model.to_dict()

    def get_interleaved_model_names(self, tasks=None) -> List[str]:
        from itertools import zip_longest

        model_names = []
        if tasks is None:
            model_names = [
                self.t2i_model_names,
                self.t2v_model_names,
                self.i2i_model_names,
                self.i2v_model_names,
            ]
        else:
            for task in tasks:
                model_names.append(self.get_model_names_by_task(task))

        shuffled_model_names = (
            random.sample(names, len(names)) for names in model_names
        )
        return [
            m
            for quad in zip_longest(*shuffled_model_names)
            for m in quad
            if m is not None
        ]

    def get_modality(self, model_name: str) -> str:
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        return "video" if model.task == ModelTask.TEXT_TO_VIDEO else "image"

    def get_task(self, model_name: str) -> str:
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        return model.task.value

    def get_output_media_type(self, model_name: str) -> str:
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        return model.media_type.value
