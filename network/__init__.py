from .Default.Model import Model as Default

models = {
    'default': Default,  # if --model is not specified

}


def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

