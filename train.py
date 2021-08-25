from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from models import load_model
from hp import load_hps
from tensorflow.keras import metrics, callbacks, optimizers
from datasets.dataset import Dataset
from plotting import plot
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def model_evaluation(test_gen):
    pred = (model.predict(test_gen) > 0.5).astype("int32")
    y_test = test_gen.labels
    print("Model Prediction:")
    print('Classification report:\n', classification_report(y_test, pred))
    print('Accuracy score:\n', accuracy_score(y_test, pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
    print("\nModel Evaluation on Test Set:")
    model.evaluate(test_gen, batch_size=200)


if __name__ == '__main__':
    hps = load_hps(dataset_dir="./covid-19/", model_name='resnet50', n_epochs=50, batch_size=1,
                   learning_rate=0.001,
                   lr_reducer_factor=0.2,
                   lr_reducer_patience=8, img_size=200, framework='keras')
    model = load_model(model_name=hps['model_name'])

    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=hps['learning_rate']),
                  metrics=METRICS)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=hps['lr_reducer_factor'],
                                            patience=hps['lr_reducer_patience'],
                                            verbose=1,
                                            min_delta=0.0001)

    model_checkpoint = callbacks.ModelCheckpoint("./best_model.h5", monitor='val_loss', save_best_only=True,
                                                 verbose=1)

    callbacks = [reduce_lr, model_checkpoint]

    if hps['framework'] == 'tensorflow':
        train_ds, val_ds, test_ds = Dataset.tensorflow_preprocess(dataset_dir=hps['dataset_dir'],
                                                                  img_size=hps['img_size'],
                                                                  batch_size=hps['batch_size'],
                                                                  train_augment=True, val_augment=False, split_size=0.3)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=hps['epochs'],
            batch_size=hps['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        plot(history)
        test_datagen = ImageDataGenerator(rescale=1 / 255)
        test_generator = test_datagen.flow_from_directory(hps['dataset_dir'] + "test/",
                                                          target_size=(hps['img_size'], hps['img_size']),
                                                          batch_size=hps['batch_size'], class_mode='binary')
        model_evaluation(test_generator)
    elif hps['framework'] == 'keras':
        train_generator, validation_generator, test_generator = Dataset.keras_preprocess(
            dataset_dir=hps['dataset_dir'] + "train/",
            img_size=hps['img_size'],
            batch_size=hps['batch_size'],
            augment=True, split_size=0.3)
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // hps['batch_size'],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // hps['batch_size'],
            epochs=hps['epochs'],
            callbacks=callbacks,
            verbose=2)
        plot(history)
        model_evaluation(test_generator)
