from models import load_model
from hp import load_hps

# from dataset import dataset_module_name

if __name__ == '__main__':
    hps = load_hps(model_name='resnet50', n_epochs=50, lr='0.001')
    model = load_model(model_name=hps['model_name'], **hps)
