import importlib

def find_model_using_name(model_name):
    """
    Import the module "models/[model_name]_model.py".
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("Missing file ", model_name, "_model.py in the models/ directory")
        exit(0)

    return model


def create_model(opt, dataset):
    """
    Create the model based on the given options
    """
    model = find_model_using_name(opt.model)
    instance = model(opt, dataset)
    print("model [%s] was created" % type(instance).__name__)
    return instance
