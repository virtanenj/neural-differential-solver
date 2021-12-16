import numpy as np

# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3

import tensorflow as tf
from tensorflow import keras


# tf.keras.backend.set_floatx('float64')


# Is this smart?
@tf.function
def f(x):
    f = neural_net(x)
    return f


def test_differential_equation(x):
    '''
    d^2f/dx^2 + 2 = 0
    '''
    with tf.GradientTape() as tape_1:
        tape_1.watch(x)
        with tf.GradientTape() as tape_2:
            tape_2.watch(x)
            y = f(x)
            dy_dx = tape_2.gradient(y, x)
        d2y_dx2 = tape_1.gradient(dy_dx, x)

    return (d2y_dx2 + 2)**2


# Could and should this be done without nested functions?
def my_loss_fn(x):
    # Boundaries
    a = x[0]
    b = x[-1]
    def loss(y_true, y_pred):
        # The boundary conditions are f(a) = 0 and f(b) = 0
        sum_over_boundary_terms = f(a)**2 + f(b)**2
        sum_over_F = tf.math.reduce_sum(tf.map_fn(test_differential_equation, x))
        return sum_over_F / i_max + sum_over_boundary_terms
    return loss


def build_model(units):
    input_layer = keras.layers.Input(shape=(1,))
    hidden_layer = keras.layers.Dense(units, activation=tf.nn.sigmoid)(input_layer)
    output_layer = keras.layers.Dense(1, activation=tf.identity)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


a = 0
b = 2
i_max = 100
epochs = 10000

x = np.linspace(a, b, i_max, dtype=np.float32)
x_tensor = tf.convert_to_tensor(x)
x_tensor = tf.reshape(x_tensor, (i_max, 1))

neural_net = build_model(units=100)
neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs)


x_test = np.linspace(a, b, 20)
y_prediction = neural_net.predict(x_test)
# The solution for d^2f/dx^2 + 2 = 0 and the boundaries
# f(0) = 0, f(2) = 0 is
# f(x) = -(x - 2)x = -x^2 + 2x
x_correct = np.linspace(a, b, 100)
y_correct = -x_correct**2 + 2 * x_correct


import matplotlib.pyplot as plt

plt.plot(x_test, y_prediction, label='The model')
plt.plot(x_correct, y_correct, label='Correct solution')
plt.grid(True)
plt.legend()
plt.show()
