import tensorflow as tf

class TensorFlowConvolutions:
    def __init__(self, features, input_shape, conv_dim=2):
        if conv_dim not in [2, 3]:
            raise ValueError("conv_dim must be 2 or 3")
        self.features = features
        self.input_shape = input_shape
        self.conv_dim = conv_dim

    def conv2d_block(self, x):
        for feat in self.features[:-1]:
            x = tf.keras.layers.Conv2D(feat, kernel_size=(3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        return tf.keras.layers.Flatten()(x)

    def conv3d_block(self, x):
        for feat in self.features[:-1]:
            x = tf.keras.layers.Conv3D(feat, kernel_size=(3, 3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        return tf.keras.layers.Flatten()(x)

    def create_model(self):
        inputs = tf.keras.Input(shape=self.input_shape[1:])
        if self.conv_dim == 2:
            x = self.conv2d_block(inputs)
        elif self.conv_dim == 3:
            x = self.conv3d_block(inputs)
        else:
            raise ValueError("conv_dim must be 2 or 3")

        for feat in self.features[:-1]:
            x = tf.keras.layers.Dense(feat, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.features[-1])(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
