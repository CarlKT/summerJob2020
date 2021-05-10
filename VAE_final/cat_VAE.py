from GumbelSoftmax import Gumbel_Softmax
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dense, Input, Activation, Reshape, Flatten
import sklearn

class VAE(Model):
    def __init__(self, latent_dim, inputShape, temperature=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inputShape = inputShape
        self.tau = temperature
        #self.encoder = Encoder(latent_dim, encoder_shape)
        self.encoder = tf.keras.Sequential(
            [
                Input(self.inputShape),
                Conv1D(8, 3, activation="relu", strides=2, padding="same"),
                Conv1D(16, 3, activation="relu", strides=2, padding="same"),
                Conv1D(48, 3, activation="relu", strides=3, padding="same"),
                Conv1D(144, 3, activation="relu", strides=3, padding="same"),
                Flatten(),
                Dense(latent_dim + latent_dim)
            ]
        )
        #self.decoder = Decoder()
        self.decoder = tf.keras.Sequential(
            [
                Input((latent_dim,)),
                Dense((14 * 144), activation="relu"),
                Reshape((14, 144)),
                Conv1DTranspose(48, 3, activation="relu", strides=3, padding="same"),
                Conv1DTranspose(16, 3, activation="relu", strides=3, padding="same"),
                Conv1DTranspose(8, 3, activation="relu", strides=2, padding="same"),
                Conv1DTranspose(8, 3, activation="relu", strides=2, padding="same"),
                Conv1DTranspose(5, 3, activation="relu", padding="same"),
                tf.keras.layers.Activation(Gumbel_Softmax(temperature=self.tau))
            ]
        )
    
    def call(self, x):
        x = self.encode(x)
        return self.decode(x)

    @tf.function    
    def full_summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(
            "Latent dimension: " + str(self.latent_dim) + "\n" +
            "Temperature: " + str(self.tau)
        )

    @tf.function
    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.sample(mean, log_var)
        return mean, log_var, z
    
    @tf.function
    def decode(self, z, hard=False):
        #self.decoder.get_layer(index=-1).hard = hard
        #reconstruction = self.decoder(z)
        #self.decoder.get_layer(index=-1).hard = False
        return self.decoder(z)
    
    @tf.function
    def sample(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def train_step(self, data):
        #if isinstance(data, tuple):
        #    data = data[0]
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            cn = tf.keras.losses.categorical_crossentropy(data, reconstruction)
            # reconstruction_loss = tf.reduce_sum(cn)
            reconstruction_loss = 504 * tf.reduce_mean(cn)
            # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(cn, axis=1))
            kl_loss = 1 + log_var - tf.square(mean) - tf.exp(log_var)
            kl_loss = -0.5 * tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

#Visualization tools toDO: add kwargs for plt.subplots fixMe: load_data might not have n elements
def plot_latent_peaks(n, model=None, sample_size=20, show_null=True, load_dir=None, save_dir=None, *plot_args, **bar_kw):
    # Check if model meets requirements
    if model != None:
        assert isinstance(model, VAE) and model.latent_dim == 2, "Model needs to be a VAE with 2 latent dimensions."

    # Initialize firgure, rng and scaling
    grid_spec = {"wspace" : 0, "hspace" : 0}
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, gridspec_kw = grid_spec, *plot_args)
    rng = np.random.default_rng()
    scale = 1.0
    line_width = 1
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    seq_avg_data = np.ndarray((n,n), dtype=np.ndarray)

    # Optionally load data
    if load_dir != None:
        seq_avg_data = np.load(load_dir, allow_pickle=True)
    
    # Get sequence information from vae and plot as a distribution
    for i, x_start in enumerate(grid_x):
        for j, y_start in enumerate(grid_y):
            print("processing grid " + str(i) + ", " + str(j))
            seq_avg = seq_avg_data[i][j]
            if load_dir == None:
                seq_avg = gen_peak(model, rng, n, x_start, y_start, sample_size=sample_size)
                seq_avg_data[i][j] = seq_avg
            seqs = split_seq(seq_avg)

            # Plot generated sequences
            for seq in seqs:
                axs[i,j].bar(range(len(seq_avg[0])), seq[0], line_width, color=colors[0], **bar_kw)
                for k in range(len(seq_avg) - 2 + int(show_null)):
                    axs[i,j].bar(range(len(seq_avg[0])), seq[k+1], line_width, color=colors[k+1], **bar_kw)
    
    # Optionally save data
    if save_dir != None:
        np.save(save_dir, seq_avg_data)
    
    fig.tight_layout()
    print("plotting...")
    plt.show()

def gen_peak(model, rng, n, x_start, y_start, sample_size=20):
    # Initial assignment of seq_sum
    z_0 = 2/n * rng.random() + x_start
    z_1 = 2/n * rng.random() + y_start
    z = np.array([[z_0, z_1]])
    seq_sum = np.transpose(model.decode(z)[0])

    # toDo change 2 to 2*scale eventually
    for i in range(sample_size-2):
        z_0 = 2/n * rng.random() + x_start
        z_1 = 2/n * rng.random() + y_start
        z = np.array([[z_0, z_1]])
        seq_sum = np.add(seq_sum, np.transpose(model.decode(z)[0]))

    return seq_sum / sample_size

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    reduced_mean = sklearn.TSNE(n_components=2).fit_transform(z_mean)
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_mean[:, 0], reduced_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("TSNE[0]")
    plt.ylabel("TSNE[1]")
    plt.show()

def split_seq(seq):
    """ Splits a sequence into (descending) sorted array of sequences while maintaining their relative position. 
    For example, [1,3,2,...] --> [..., [0,3,0,0,0], [0,0,2,0,0], [1,0,0,0,0]]"""
    split_seq = np.array([np.zeros(seq.shape) for i in range(seq.shape[0])])
    for n in range(seq.shape[1]):
        sorted_ind = np.argsort(seq[:, n])
        for i, ind in enumerate(reversed(sorted_ind)):
            split_seq[i, ind, n] = seq[ind, n]
    return split_seq

if __name__ == "__main__":
    load_dir = "/home/ctessier/tba/neural_nets/VAE_final/seq_avg.npy"
    seq_arr = np.load(load_dir, allow_pickle=True)
    seq = seq_arr[0][0]
    # print(seq[:,0])
    # print([i for i in reversed(np.argsort(seq[:,0]))])
    print(split_seq(seq)[1])