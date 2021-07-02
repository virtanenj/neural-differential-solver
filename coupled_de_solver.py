import numpy as np
# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3
import tensorflow as tf
from tensorflow import keras

# tf.keras.backend.set_floatx('float64')


# ?????
@tf.function
def f1(x):
    return tf.reshape(neural_net(x)[:, 0], (len(x), 1))

@tf.function
def f2(x):
    return tf.reshape(neural_net(x)[:, 1], (len(x), 1))


def coupled_diff_eq_1(x):
    '''
    df/dx-cos(x)-f(x)^2-g(x)+1+x^2+sin^2(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1 = f1(x)
        dy1_dx = tape.gradient(y1, x)
    y2 = f2(x)
    return (dy1_dx - tf.math.cos(x) - f1(x)**2 - f2(x) + x**2 + tf.math.sin(x)**2)**2


def coupled_diff_eq_2(x):
    '''
    dg/dx-2x+(1+x^2)sin(x)-f(x)g(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y2 = f2(x)
        dy2_dx = tape.gradient(y2, x)
    y1 = f1(x)
    return (dy2_dx - 2 * x + (1 + x**2) * tf.math.sin(x) - f1(x) * f2(x))**2


def my_loss_fn(x):
    def loss(y_true, y_pred):
        # L = (1/i_max)(F_1^2 + F_2^2) + (f_1(0) - 0)^2 + (f_2(0) - 1)^2
        sum_over_F1 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_1, x))
        sum_over_F2 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_2, x))
        sum_over_boundary_terms = f1(x_b)**2 + (f2(x_b) - 1)**2
        return (sum_over_F1 + sum_over_F2) / i_max + sum_over_boundary_terms
    return loss


def build_model(units):
    input_layer = keras.layers.Input(shape=(1,))
    hidden_layer = keras.layers.Dense(units, activation=tf.nn.sigmoid)(input_layer)
    output_layer = keras.layers.Dense(2, activation=tf.identity)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


a = 0
b = 2
i_max = 100
units = 10
epochs = 1000

x = np.linspace(a, b, i_max, dtype=np.float32)
x_tensor = tf.reshape(tf.convert_to_tensor(x), (i_max, 1))
x_b = tf.reshape(tf.convert_to_tensor(0), (1, 1))


neural_net = build_model(units=units)
neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs)


x_test = np.linspace(a, b, 100)
y_predictions = neural_net.predict(x_test)
y1_prediction = y_predictions[:, 0]
y2_prediction = y_predictions[:, 1]


# y1_correct = ???
# y2_correct = ???


import matplotlib.pyplot as plt

plt.plot(x_test, y1_prediction, label='y1', marker='o')
plt.plot(x_test, y2_prediction, label='y2', marker='o')
# plt.plot(x_correct, y_correct, label='Correct solution')
plt.legend()
plt.show()
