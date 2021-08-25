from .resnet50 import Resnet50
from .InceptionResNetV2 import InceptionResnetV2

MODELS = dict(resnet50=Resnet50,
              inception_resnetv2=InceptionResnetV2
              # other models
              )


def load_model(model_name, **kwargs):
    return MODELS[model_name](**kwargs).get_model()
