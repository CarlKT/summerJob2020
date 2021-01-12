##IMPORTS##
import numpy as np
import tensorflow as tf
import time
from joblib import Parallel, delayed

#Project sources:
#1-https://keras.io/examples/generative/vae/
#2-https://www.tensorflow.org/tutorials/generative/cvae
#3-https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

#MNIST is regression while chip-seq is not
#Needs to have a validation split


class Seq_Embeding(tf.keras.layers.Layer):
    """Used to turn genomic sequences into binary classification. Not useful as
    this object lacks flexibility and it is about 10 times slower than its
    alternative. It is also not necessary to 'smartly' generate an embedding
    dict as genome sequences have small dimensionality."""
    def __init__(self):
        super(Seq_Embeding, self).__init__()

    def dict_generator(self, input):
        embeding_dict = {}

        dimension = len(np.unique(input))
        iterator = zip(range(dimension), np.unique(input))

        for (index, variable) in iterator:
            zeros = [0.0] * dimension
            zeros[index] = 1.0
            embeding_dict[variable] = zeros

        return embeding_dict

    def embeder(self, embeding_dict, seq):
        embeded_seq = [embeding_dict[variable] for variable in seq]
        return embeded_seq

    def call(self, input, n_jobs=1):
        if type(input) != list:
            raise TypeError('input must be a list. Got %s instead' %
                            type(input))
        #Generate embeding dictionary
        embeding_dict = self.dict_generator(input)
        embeded_input = Parallel(n_jobs=n_jobs)(
            delayed(self.embeder)(embeding_dict, seq) for seq in input)
        embeded_seq = np.array(embeded_input)

        self.embeded_shape = embeded_seq.shape
        return embeded_seq


#OPTION 1
def binarize_sequence(sequence):
    binary_base = {
        'C': [1.0, 0.0, 0.0, 0.0],
        'G': [0.0, 1.0, 0.0, 0.0],
        'A': [0.0, 0.0, 1.0, 0.0],
        'T': [0.0, 0.0, 0.0, 1.0]
    }
    binarized_seq = [binary_base[base] for base in sequence]
    return binarized_seq


def gen_rand_seq(length):
    return [bases[np.random.randint(0, high=4)] for i in range(length)]


if __name__ == "__main__":

    bases = ['C', 'G', 'A', 'T']
    seq_list = Parallel(n_jobs=1)(delayed(gen_rand_seq)(504)
                                  for i in range(1000))

    #OPTION 1
    start1 = time.time()
    binarized_seq = Parallel(n_jobs=1)(delayed(binarize_sequence)(seq)
                                       for seq in seq_list)
    print(time.time() - start1)

    #OPTION 2

    embeder = Seq_Embeding()
    start2 = time.time()
    binarized_seq = embeder(seq_list, n_jobs=1)
    print(time.time() - start2)
    embeder.embeded_shape
