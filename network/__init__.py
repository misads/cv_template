from .AOD.Model import Model as AOD
from .FFA.Model import Model as FFA


models = {
    'AOD': AOD,
    'FFA': FFA,  # --model MUST be specified now
}


def get_model(model: str):
    if model is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model in models:
        return models[model]
    else:
        raise AttributeError('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))
