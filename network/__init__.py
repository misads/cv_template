"""
现在network目录中的模型将会自动检测，不再需要手动导入
"""
import os
import importlib

model_names = os.listdir('network')


models = {}

for name in model_names:
    if os.path.isdir(f'network/{name}'):
        if name == '__pycache__':
            continue

        models[name] = importlib.import_module(f'.{name}.Model', 'network').Model


def get_model(model: str):
    if model is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model in models:
        return models[model]
    else:
        raise AttributeError('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))
