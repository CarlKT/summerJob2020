###Imports###
import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model


def create_class_model(num_inputs, num_outputs, hidden_shape, summary=True):
    """Creates a classification model with any number of inputs, outputs. Additionaly,
        hidden_shape configures hidden layer architecture, with each element of
        hidden_layer represents the number of nodes in that layer"""

    input_tensor = Input(shape=(num_inputs, ))
    tensors = [input_tensor]

    #Create hidden layers
    for i in range(len(hidden_shape)):
        num_nodes = hidden_shape[i]
        hidden_tensor_0 = tensors[i]
        tensor_name = 'hidden_layer_%s' % (i + 1)
        hidden_tensor_1 = Dense(num_nodes, activation='relu',
                                name=tensor_name)(hidden_tensor_0)
        tensors.append(hidden_tensor_1)

    output_tensor = Dense(num_outputs, activation='softmax',
                          name='output')(tensors[-1])

    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    if summary:
        print(model.summary())
    return model


#def train_model(predictors, labels, model):
features, labels = load_iris(return_X_y=True)
df = pd.DataFrame(features)
df['labels'] = labels

num_inputs = features.shape[1]
num_outputs = len(np.unique(labels))
hidden_shape = [100, 100]
model = create_class_model(num_inputs, num_outputs, hidden_shape, summary=True)

#model.fit(df[[0, 1, 2, 3]], df['labels'], epochs=20, validation_split=0.2)
