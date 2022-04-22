from models.resnet import resnet18
from models.vgg import vgg13, vgg16

MODELS = {
    'resnet18': resnet18,
    'vgg13': vgg13,
    'vgg16': vgg16,
}


def build_model(model_name, num_classes):
    assert model_name in MODELS.keys()
    model = MODELS[model_name](num_classes)
    return model
