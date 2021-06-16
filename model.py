import numpy as np

# Disables some warnings (TensorFlow gives a lot of them)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3

import tensorflow as tf
from tensorflow import keras


# What is the point of this function?
# Why can you call neural_net() 
def f(x):
    f = neural_net(x)
    return f


def differential_equation(x):
    '''The equation which is to be solved. Have this in
    the form of F(x, x', ... x^(n)) = 0.'''
    pass


# ????
def boundary(b):
    '''The boundary term for the given equation. 
    b is the boundary or list/array of boundaries.'''
    pass


# What is the point of using nested/inner functions?
def my_loss_fn(x):
    def loss(y_true, y_pred):
        sum_over_F = tf.math.reduce_sum(tf.map_fn(differential_equation, x))
        sum_over_boundary_terms = 0  # ???
        return sum_over_F / i_max + sum_over_boundary_terms
    return loss


def build_model(units):
    input_layer = keras.layers.Input(shape=(1,))
    # For now just 1 hidden layer
    hidden_layer = keras.layers.Dense(units, activation=tf.nn.sigmoid)(input_layer)
    output_layer = keras.layers.Dense(1, activation=tf.identity)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def train_model(x_train, batchs, epochs):
    pass


