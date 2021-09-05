"""
This module contains hyper-parameter modules
"""


def load_hps(dataset_dir, model_name, batch_size, n_epochs, learning_rate, lr_reducer_factor, lr_reducer_patience, img_size, framework, **kwargs) -> dict:
    hps = dict(
        dataset_dir=dataset_dir, model_name=model_name, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
        lr_reducer_factor=lr_reducer_factor,
        lr_reducer_patience=lr_reducer_patience, img_size=img_size, framework=framework,
    )
    if kwargs is not None:
        hps.update(kwargs)
    return hps
