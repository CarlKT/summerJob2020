import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv2da = Conv2D(32,
                              3,
                              activation="relu",
                              strides=2,
                              padding="same")
        self.conv2db = Conv2D(64,
                              3,
                              activation="relu",
                              strides=2,
                              padding="same")
        self.flatten = Flatten()
        self.encoded_vect = Dense(16, activation="relu")
        self.z_mean = Dense(latent_dim, name="z_mean")
        self.z_log_var = Dense(latent_dim, name="z_log_var")

    def call(self, encoder_inputs):
        x = self.conv2da(encoder_inputs)
        x = self.conv2db(x)
        x = self.flatten(x)
        x = self.encoded_vect(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.d1 = Dense(7 * 7 * 64, activation="relu")
        self.reshape = Reshape((7, 7, 64))
        self.conv2dTa = Conv2DTranspose(64,
                                        3,
                                        activation="relu",
                                        strides=2,
                                        padding="same")
        self.conv2dTb = Conv2DTranspose(32,
                                        3,
                                        activation="relu",
                                        strides=2,
                                        padding="same")
        self.decoder_outputs = Conv2DTranspose(1,
                                               3,
                                               activation="sigmoid",
                                               padding="same")

    def call(self, latent_inputs):
        x = self.d1(latent_inputs)
        x = self.reshape(x)
        x = self.conv2dTa(x)
        x = self.conv2dTb(x)
        return self.decoder_outputs(x)


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_inputs = Input(shape=(latent_dim, ))
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def call(self, encoder_inputs):
        x = self.encoder(encoder_inputs)
        return self.decoder(x)

    def full_summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":
    encoder_inputs = Input(shape=(28, 28, 1))
    latent_inputs = Input(shape=(2, ))

    vae = VAE(latent_dim=2)
    vae(encoder_inputs)
    vae.full_summary()

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#    vae.compile(optimizer=keras.optimizers.Adam())
#    vae.fit(mnist_digits, epochs=30, batch_size=128)
