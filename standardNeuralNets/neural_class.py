###Imports###
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class MyInput(Model):
    """Same as neural_network_test.py but object oriented. DISCLAIMER: mostly
       inspired by tensorflow documentation."""
    def __init__(self):
        super(MyInput, self).__init__()
        #self.input = Input(32, 3, activation='relu')
        #self.d1 = Dense(128, activation='relu')
        #self.d2 = Dense(10)
        #self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[0]),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):

        input_shape = tf.matmul(inputs, self.w) + self.b
        self.num_inputs = inputs.shape[0]
        self.units = inputs.shape[1]
        return input_shape


class ModelBuilder():
    def __init__(self,
                 inputs,
                 outputs,
                 prediction_type='categorical',
                 hidden_shape=None):
        self.inputs = inputs
        self.outputs = outputs
        self.prediction_type = prediction_type
        self.hidden_shape = hidden_shape
        self.model = Model()
        self.splitter = train_test_split(inputs,
                                         outputs,
                                         test_size=0.1,
                                         random_state=None)

    def create(self, summary=True):
        num_inputs = self.inputs.shape[1]
        if self.prediction_type == 'categorical':
            num_outputs = 1
        else:
            raise ValueError('Other prediction types not yet supported.')

        input_tensor = Input(shape=(num_inputs, ))
        tensors = [input_tensor]

        #Create hidden layers
        for i in range(len(self.hidden_shape)):
            num_nodes = self.hidden_shape[i]
            tensor_1_name = 'hidden_layer_%s' % (i + 1)

            tensor_0 = tensors[i]
            tensor_1 = Dense(num_nodes, activation='relu',
                             name=tensor_1_name)(tensor_0)
            tensors.append(tensor_1)

        output_tensor = Dense(num_outputs, activation='softmax',
                              name='output')(tensors[-1])
        optimizer = Adam(learning_rate=0.001)
        self.model = Model(input_tensor, output_tensor)
        self.model.compile(optimizer=optimizer,
                           loss='mean_absolute_error',
                           metrics=['accuracy'])
        if summary:
            print(self.model.summary())
        return self.model

    def fit_builder(self, **options):
        x_train, x_test, y_train, y_test = self.splitter
        self.model.fit(x_train, y_train, **options)


if __name__ == "__main__":

    #Load data
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #inputs = MyInput()

    #x = tf.ones((2, 2))
    #inputs = model(x_train)
    #print(inputs)
    inputs, outputs = load_iris(return_X_y=True)
    builder = ModelBuilder(inputs, outputs)
    builder.hidden_shape = [150, 150]
    model = builder.create(summary=True)
    builder.fit_builder(epochs=20, validation_split=0.1)
