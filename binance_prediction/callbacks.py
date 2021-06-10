#
#
#

def get_callbacks(test_generator):
    import tensorflow as tf
    from tensorflow.python.keras.callbacks import Callback

    class Metrics(Callback):
        def __init__(self, test_generator):
            Callback.__init__(self)
            self.x, self.y = test_generator.__next__()

        def on_batch_end(self, batch, logs=None):
            val_pred = self.model.predict_on_batch(self.x[0:3])
            print('Pred: ', val_pred * 100)
            print('Y: ', self.y[0:3] * 100)

        # def on_epoch_end(self, epoch, logs=None):
        #     val_pred = self.model.predict_on_batch(self.x[0:3])
        #     print('Pred: ', val_pred)
        #     print('Y: ', self.y[0:3])

    metrics = Metrics(test_generator)
    return [tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0), metrics]
