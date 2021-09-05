import matplotlib.pyplot as plt


def plot(history):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('query', )
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('query')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
