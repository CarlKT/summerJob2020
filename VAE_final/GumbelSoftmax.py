import tensorflow as tf

class Gumbel_Softmax(tf.keras.layers.Layer):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
      Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
      Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
    def __init__(self, temperature, hard=False):
        super(Gumbel_Softmax, self).__init__()
        self.temperature = temperature
        self.hard = hard
        
    def call(self, input):
        activated_input = self.gumbel_softmax(input, self.temperature)
        out_tensor = tf.convert_to_tensor(activated_input)
        return out_tensor

    def sample_gumbel(self, shape, eps=1.0e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape,
                              maxval=1.,
                              dtype=tf.dtypes.float32)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    def gumbel_softmax(self, logits, temperature):
        y = self.gumbel_softmax_sample(logits, temperature)
        if self.hard:
            k = tf.shape(logits)[-1]
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)),
                             y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y