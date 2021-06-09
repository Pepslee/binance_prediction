def build_model(shape, loss_fn):
    import tensorflow as tf
    print(tf.__version__)
    regularizer = tf.keras.regularizers.l2(0.01)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=shape),
        tf.keras.layers.Conv1D(5, 3, strides=1, kernel_regularizer=regularizer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Conv2D(4, (5, 3), strides=(1, 1), kernel_regularizer=regularizer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Conv2D(12, (5, 3), strides=(1, 1), kernel_regularizer=regularizer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Conv2D(32, (5, 3), kernel_regularizer=regularizer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #               loss=loss_fn,
    #               metrics=[tf.keras.metrics.Recall(thresholds=0.2, name='recall_0.2'),
    #                        tf.keras.metrics.Recall(thresholds=0.4, name='recall_0.4'),
    #                        tf.keras.metrics.Recall(thresholds=0.6, name='recall_0.6'),
    #                        tf.keras.metrics.Precision(thresholds=0.2, name='precision_0.2'),
    #                        tf.keras.metrics.Precision(thresholds=0.4, name='precision_0.4'),
    #                        tf.keras.metrics.Precision(thresholds=0.6, name='precision_0.6')]
    #               )
    return model
