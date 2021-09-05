"""
This module contains preprocessing and augmentation modules
"""

import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from preparing_datasets import PreparingDatasets


class Dataset:
    @staticmethod
    def tensorflow_preprocess(dataset_dir="../covid-19/", img_size=200, batch_size=32,
                              train_augment=True, val_augment=False, split_size=0.3):
        # Path to your dataset directory
        #  After Preparing it should be like this:
        #     covid-19/..
        #         /train/..
        #             /Normal/
        #             /Covid/
        #         /val/..
        #             /Normal/
        #             /Covid/
        #         /test/..
        #             /Normal/
        #             /Covid/

        preparing_datasets = PreparingDatasets(framework='tensorflow')
        preparing_datasets.preparing_datasets(split_size=split_size)
        builder = tfds.folder_dataset.ImageFolder(root_dir=dataset_dir)
        train_ds = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
        val_ds = builder.as_dataset(split='val', as_supervised=True)
        test_ds = builder.as_dataset(split='test', as_supervised=True)

        AUTOTUNE = tf.data.AUTOTUNE

        resize_and_rescale = tf.keras.Sequential([
            preprocessing.Resizing(img_size, img_size),
            preprocessing.Rescaling(1. / 255)
        ])

        data_augmentation = tf.keras.Sequential([
            preprocessing.RandomZoom(0.2),
            preprocessing.RandomFlip("horizontal"),
        ])

        def prepare(ds, shuffle=False, augment=False):
            # Resize and rescale all datasets
            ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                        num_parallel_calls=AUTOTUNE)

            if shuffle:
                ds = ds.shuffle(1000)

            # Batch all datasets
            ds = ds.batch(batch_size)

            # Use data augmentation only on the training set
            if augment:
                ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)

            # Use buffered prefetching on all datasets
            return ds.prefetch(buffer_size=AUTOTUNE)

        train_ds = prepare(train_ds, shuffle=True, augment=train_augment)
        val_ds = prepare(val_ds, augment=val_augment)
        test_ds = prepare(test_ds)
        return train_ds, val_ds, test_ds

    @staticmethod
    def keras_preprocess(dataset_dir="../covid-19/", img_size=200, batch_size=32, augment=True, split_size=0.3):

        # Path to your dataset directory
        #  After Preparing it should be like this:
        #     covid-19/..
        #         /train/..
        #             /Normal/
        #             /Covid/
        #         /test/..
        #             /Normal/
        #             /Covid/

        preparing_datasets = PreparingDatasets(framework='keras')
        preparing_datasets.preparing_datasets()
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.3)
        else:
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                validation_split=split_size)

        test_datagen = ImageDataGenerator(rescale=1 / 255)

        train_generator = train_datagen.flow_from_directory(
            dataset_dir + "train/",
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            subset='training')

        validation_generator = train_datagen.flow_from_directory(
            dataset_dir + "train/",
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            subset='validation')

        test_generator = test_datagen.flow_from_directory(dataset_dir + "test/", target_size=(img_size, img_size),
                                                          batch_size=batch_size, class_mode='binary')

        return train_generator, validation_generator, test_generator
