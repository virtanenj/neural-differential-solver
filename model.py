import numpy as np

# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3

import tensorflow as tf
from tensorflow import keras


# tf.keras.backend.set_floatx('float64')


# IS this smart?
@tf.function
def f(x):
    f = neural_net(x)
    return f



def differential_equation(x):
    '''
    The equation which is to be solved. Have this in
    the form of F(x, x', ... x^(n)) = 0.

    This function implements d^2f/dx^2 + df/dx - 10 = 0.
    '''

    with tf.GradientTape() as tape_1:
        tape_1.watch(x)
        with tf.GradientTape() as tape_2:
            tape_2.watch(x)
            y1 = f(x)
            dy_dx = tape_2.gradient(y1, x)
        d2y_dx2 = tape_1.gradient(dy_dx, x)

    with tf.GradientTape() as tape_3:
        tape_3.watch(x)
        y2 = f(x)
        dy_dx = tape_3.gradient(y2, x)

    return (d2y_dx2 + dy_dx - 10)**2


# ????
def boundary(b):
    '''The boundary term for the given equation. 
    b is the boundary or list/array of boundaries.'''
    pass


# Could and should this be done without nested functions?
def my_loss_fn(x):
    # Boundaries
    a = x[0]
    b = x[-1]
    def loss(y_true, y_pred):
        boundaries = tf.convert_to_tensor(np.array([a, b]))

        # The boundary conditions are f(a) = 0 and f(b) = 0
        # sum_over_boundary_terms = tf.math.reduce_sum(tf.map_fn(f, boundaries))  # ???

        sum_over_boundary_terms = f(a)**2 + f(b)**2
        sum_over_F = tf.math.reduce_sum(tf.map_fn(differential_equation, x))

        return sum_over_F / i_max + sum_over_boundary_terms
    return loss


def build_model(units):
    input_layer = keras.layers.Input(shape=(1,))
    # For now just 1 hidden layer
    hidden_layer = keras.layers.Dense(units, activation=tf.nn.sigmoid)(input_layer)
    output_layer = keras.layers.Dense(1, activation=tf.identity)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


# def build_and_train(a, b, i_max, epochs, units):
#     x = np.linspace(a, b, i_max)
#     x_tensor = tf.convert_to_tensor(x)
#     x_tensor = tf.reshape(x_tensor, (i_max, 1))
#     neural_net = build_model(units=units)
#     neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
#     neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs)
#     return neural_net

a = 0
b = 2
i_max = 100
epochs = 2000  # 3000 <= gives a good result


# model = build_and_train(a, b, i_max, epochs, units)


x = np.linspace(a, b, i_max, dtype=np.float32)
x_tensor = tf.convert_to_tensor(x)
x_tensor = tf.reshape(x_tensor, (i_max, 1))

neural_net = build_model(units=100)
neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')

neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs)



x_test = np.linspace(a, b, i_max)
y_prediction = neural_net.predict(x_test)

# The correct solution for d^2f/dx^2 + df/dx - 10 = 0 with the
# domain 0<=x<=2 and boundary conditions f(0) = 0, f(2) = 0 is
# f(x) = 20/(1-Exp(-2)) Exp(-x) - 20/(1-Exp(-2)) + 10x
y_correct = 20 / (1 - np.exp(-2)) * np.exp(-x_test) - 20 / (1 - np.exp(-2)) + 10 * x_test


import matplotlib.pyplot as plt

plt.plot(x_test, y_prediction, label='The model')
plt.plot(x_test, y_correct, label='Correct solution')
plt.legend()
plt.show()

