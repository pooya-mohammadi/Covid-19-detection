"""
This Module contains customized model
"""
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model


class InceptionResnetV2:
    def __init__(self, img_w=200, img_h=200, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self) -> Model:
        # using InceptionResNetV2
        INPUT_SHAPE = self.input_shape

        # get the pretrained model
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=INPUT_SHAPE,
                                                             include_top=False,
                                                             weights='imagenet')
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
