import matplotlib.pyplot as plt
import io


def gen_plot(x, title):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.title(title)
    for i in range(len(x)):
        plt.plot(range(len(x[i])), x[i] * 100, label=str(i))
    plt.legend()
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def get_callbacks(test_generator):
    import tensorflow as tf
    from tensorflow.python.keras.callbacks import Callback

    class Metrics(Callback):
        def __init__(self, test_generator):
            Callback.__init__(self)
            self.x, self.y = test_generator.__next__()

        def on_batch_end(self, batch, logs=None):
            writer = tf.summary.create_file_writer('logs')
            # print(self.x[0:3])
            plot_buf_pred = gen_plot(self.model.predict_on_batch(self.x[0:3]), 'pred')
            plot_buf_y = gen_plot(self.y[0:3], 'Y')
            # Convert PNG buffer to TF image
            image_pred = tf.image.decode_png(plot_buf_pred.getvalue(), channels=4)
            image_y = tf.image.decode_png(plot_buf_y.getvalue(), channels=4)
            # Add the batch dimension
            image_y = tf.expand_dims(image_y, 0)
            image_pred = tf.expand_dims(image_pred, 0)
            image = tf.concat(axis=0, values=[image_y, image_pred])
            with writer.as_default():
                tf.summary.image('test_pred', image, step=batch)
                # tf.summary.image('candles1', image, step=batch)
                writer.flush()
            # tf.summary.write('logs')
            # val_pred = self.model.predict_on_batch(self.x[0:3])
            # print('Pred: ', val_pred * 100)
            # print('Y: ', self.y[0:3] * 100)

        # def on_batch_end(self, batch, logs=None):
        #     val_pred = self.model.predict_on_batch(self.x[0:3])
        #     print('Pred: ', val_pred * 100)
        #     print('Y: ', self.y[0:3] * 100)

    metrics = Metrics(test_generator)
    return [tf.keras.callbacks.TensorBoard(log_dir='logs', write_images=True, profile_batch=0), metrics]
