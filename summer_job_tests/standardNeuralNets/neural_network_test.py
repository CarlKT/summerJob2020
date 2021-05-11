###Imports###
import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping


def create_class_model(num_inputs, num_outputs, hidden_shape, summary=True):
    """Creates a classification model with any number of inputs, outputs. Additionaly,
        hidden_shape configures hidden layer architecture, with each element of
        hidden_shape representing the number of nodes in that layer"""

    input_tensor = Input(shape=(num_inputs, ))
    tensors = [input_tensor]

    #Create hidden layers
    for i in range(len(hidden_shape)):
        num_nodes = hidden_shape[i]
        tensor_1_name = 'hidden_layer_%s' % (i + 1)

        tensor_0 = tensors[i]
        tensor_1 = Dense(num_nodes, activation='relu',
                         name=tensor_1_name)(tensor_0)
        tensors.append(tensor_1)

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
#features_df = pd.DataFrame(features)
#binary_labels = pd.get_dummies(labels, prefix='label', dtype='float')
#df = pd.concat([features_df, binary_labels], axis=1)
#print(df.head())
df = pd.DataFrame(features)
df['labels'] = labels

num_inputs = features.shape[1]
#num_outputs = len(np.unique(labels))
num_outputs = 1
hidden_shape = [100, 100]

model = create_class_model(num_inputs, num_outputs, hidden_shape, summary=True)
"""
model.fit(df[[0, 1, 2, 3]],
          df[['label_0', 'label_1', 'label_2']],
          epochs=20,
          validation_split=0.2,
          callbacks=EarlyStopping(patience=5))

model.predict(df[[0, 1, 2, 3]], df[['label_0', 'label_1', 'label_2']])
"""
model.fit(df[[0, 1, 2, 3]],
          df['labels'],
          epochs=20,
          validation_split=0.2,
          callbacks=EarlyStopping(patience=5))

model.predict(df[[0, 1, 2, 3]], df[['labels']])
