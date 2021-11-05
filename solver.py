
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.activations import sigmoid


# What about custom activation functions (like sin-function)??

class Solver():
    def __init__(self, i_max, domain, my_diff_fns, my_boundary_terms, xbs, hidden_layers, layers):
        # What about multidimensional input??
        self.i_max = i_max
        self.x = np.linspace(domain[0], domain[1], i_max)
        self.x_tensor = tf.reshape(tf.convert_to_tensor(self.x, dtype=tf.float32), (self.i_max, 1))
        self.nn = None
        self.build_model(hidden_layers=hidden_layers, layers=layers)
        self.my_diff_fns = my_diff_fns
        self.my_boundary_terms = my_boundary_terms
        self.xbs = xbs
        self.xb_tensors = []
        for xb in xbs:
            xb_tensor = tf.reshape(tf.convert_to_tensor(xb, dtype=tf.float32), (1, 1))
            self.xb_tensors.append(xb_tensor)

    def build_model(self, hidden_layers, layers):
        # What about different activation functions??
        input_layer = Input(shape=(1,))
        prev_layer = Dense(layers[0], activation=sigmoid)(input_layer)
        for i in range(1, hidden_layers):
            hi_layer = Dense(layers[i], activation=sigmoid)(prev_layer)
            prev_layer = hi_layer
        final_layer = Dense(len(self.my_diff_fns), activation=tf.identity)(hi_layer)
        self.nn = keras.Model(inputs=input_layer, outputs=final_layer)

    def diff_terms(self, x, my_diff_fn):
        return my_diff_fn(x=x, f=self.nn)**2

    def boundary_terms(self, xb, my_boundary_term):
        return my_boundary_term(xb=xb, f=self.nn)**2

    def my_loss(self):
        def loss(y_true, y_pred):
            diff_sum = 0
            for my_diff_fn in self.my_diff_fns:
                def diff_term(x):
                    return self.diff_terms(x=x, my_diff_fn=my_diff_fn)
                diff_sum += tf.math.reduce_sum(tf.map_fn(diff_term, self.x_tensor))
            boundary_term_sum = 0
            for i, my_boundary_term in enumerate(self.my_boundary_terms):
                boundary_term_sum += self.boundary_terms(xb=self.xb_tensors[i], my_boundary_term=my_boundary_term)
            return diff_sum / self.i_max + boundary_term_sum
        return loss

    def compile_model(self):
        self.nn.compile(loss=self.my_loss(), optimizer='adam')

    def train_model(self, epochs):
        self.nn.fit(x=self.x_tensor, y=self.x_tensor, epochs=epochs, batch_size=self.i_max)

    def predict(self, x):
        return self.nn(x)


# Example function(s)
# ----------
def boundary_term1(xb, f):
    ''' f(0) = 0 '''
    return f(xb)

xb1 = 0

def boundary_term2(xb, f):
    ''' f(2) = 2 '''
    return f(xb) - 2

xb2 = 2

boundary_terms = (boundary_term1, boundary_term2)
xbs = [xb1, xb2]

def my_diff_fn(x, f):
    ''' f'(x) - x^2 = 0 '''
    with tf.GradientTape() as tape:
        tape.watch(x)  # x has to be tf tensor
        dy_dx = tape.gradient(f(x), x)
        # d2y_dx2 = tape.gradient(dy_dx(x), x)
    return dy_dx - x**2

my_diff_fns = [my_diff_fn]

# try with more functions
# ----------


# Example trainig setup
domain = (0, 2)
i_max = 100
hidden_layers = 2
layers = [100, 100]
s = Solver(i_max, domain, my_diff_fns, boundary_terms, xbs, hidden_layers, layers)
s.compile_model()
epochs = 800
s.train_model(epochs=epochs)
x = np.linspace(0, 2, 20)
# x_tensor = tf.reshape(tf.convert_to_tensor(x), (20, 1))
y = s.predict(x)

# y_correct = x**3 / 3


import matplotlib.pyplot as plt


plt.plot(x, y, label='NN')
# plt.plot(x, y_correct, label='Correct')
plt.grid(True)
plt.legend()
plt.show()
