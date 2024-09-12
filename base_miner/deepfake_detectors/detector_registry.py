import os
import importlib
import inspect

class Registry(object):
    def __init__(self):
        self.data = {}
    
    def register_module(self, module_name=None):
        def _register(cls):
            name = module_name
            if module_name is None:
                name = cls.__name__
            self.data[name] = cls
            return cls
        return _register
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # Naive implementation of automatic registration of all subclasses in given directory
    def register_all_detectors(self, base_class, module_dir):
        module_dir_path = os.path.dirname(os.path.abspath(module_dir))
        for filename in os.listdir(module_dir_path):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]
                module = importlib.import_module(f'{module_dir}.{module_name}')

                # Iterate over all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, base_class) and obj != base_class:
                        # Register the subclass
                        self.register_module()(obj)
    
DETECTOR_REGISTRY = Registry()