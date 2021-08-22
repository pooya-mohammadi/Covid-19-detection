"""
This module contains hyper-parameter modules
"""


def load_hps(model_name, n_epochs, **kwargs) -> dict:
    hps = dict(
        model_name=model_name, n_epochs=n_epochs
    )
    if kwargs is not None:
        hps.update(kwargs)
    return hps
