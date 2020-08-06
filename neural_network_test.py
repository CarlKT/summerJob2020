###Imports###
import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from tf.keras.layers import Dense, Input
from tf.keras import Model

features, labels = load_iris(return_X_y=True)
features_df = pd.DataFrame(features)

predictors = {
    column_name: list(column_data)
    for column_name, column_data in features_df.iteritems()
}
print(predictors[0])


def create_class_model(num_inputs, num_outputs):

    input_tensor = Input(shape=(num_inputs, ))
    output_tensor = Dense(num_outputs, activation='softmax')(input_tensor)

    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])

    return model


num_inputs = range(predictors.keys())
num_outputs = np.unique(labels)
model = create_class_model(num_inputs, num_outputs)

print(model.summarize())
