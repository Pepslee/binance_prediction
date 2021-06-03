from tensorflow.python.keras.callbacks import Callback


class Metrics(Callback):
    def __init__(self, test_generator):
        Callback.__init__(self)
        self.x, self.y = test_generator.__next__()

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict_on_batch(self.x)
        print(val_pred)
