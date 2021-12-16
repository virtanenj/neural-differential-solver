
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.activations import sigmoid


# What about custom activation functions (like sin-function)?
# What about multidimensional input? 
class Solver():
    def __init__(self, i_max, domain, my_diff_fns, my_boundary_terms, xbs, hidden_layers, layers, activation):
        self.i_max = i_max
        self.x = np.linspace(domain[0], domain[1], i_max)
        self.x_tensor = tf.reshape(tf.convert_to_tensor(self.x, dtype=tf.float32), (self.i_max, 1))
        self.my_diff_fns = my_diff_fns
        self.my_boundary_terms = my_boundary_terms
        self.xbs = xbs
        self.xb_tensors = []
        for xb in xbs:
            xb_tensor = tf.reshape(tf.convert_to_tensor(xb, dtype=tf.float32), (1, 1))
            self.xb_tensors.append(xb_tensor)
        self.nn = None
        # Note: the activation function is now same in every layer
        self.build_model(hidden_layers=hidden_layers, layers=layers, activation=activation)
        self.fs = []  # This is for supporting many function solutions
        for i in range(len(my_diff_fns)):
            fi = tf.reshape(self.nn(self.x_tensor)[:, i], (len(self.x), 1))
            self.fs.append(fi)

    def build_model(self, hidden_layers, layers, activation):
        # What about different activation functions??
        input_layer = Input(shape=(1,))
        prev_layer = Dense(layers[0], activation=activation)(input_layer)
        for i in range(1, hidden_layers):
            hi_layer = Dense(layers[i], activation=activation)(prev_layer)
            prev_layer = hi_layer
        final_layer = Dense(len(self.my_diff_fns), activation=tf.identity)(prev_layer)
        self.nn = keras.Model(inputs=input_layer, outputs=final_layer)

    # def diff_terms(self, x, my_diff_fn):
    #     return my_diff_fn(x=x, f=self.nn)**2

    # def boundary_terms(self, xb, my_boundary_term):
    #     return my_boundary_term(xb=xb, f=self.nn)**2

    def diff_terms(self, x, my_diff_fn):
        return my_diff_fn(x=x, fs=self.fs)**2

    def boundary_terms(self, xb, my_boundary_term):
        return my_boundary_term(xb=xb, fs=self.fs)**2

    # ???
    # def fi(self, x, i):
    #     return

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

    def compile_model(self, optimizer):
        self.nn.compile(loss=self.my_loss(), optimizer=optimizer)

    def train_model(self, epochs):
        self.nn.fit(x=self.x_tensor, y=self.x_tensor, epochs=epochs, batch_size=self.i_max)

    def predict(self, x):
        return self.nn(x)


# # Example function 1
# def boundary_term1(xb, f):
#     ''' f(0) = 0 '''
#     return f(xb)

# xb1 = 0

# def boundary_term2(xb, f):
#     ''' f(2) = 2 '''
#     return f(xb)

# xb2 = 2

# boundary_terms = (boundary_term1, boundary_term2)
# xbs = [xb1, xb2]

# def my_diff_fn(x, f):
#     ''' f''(x) + 2 = 0 '''
#     with tf.GradientTape() as tape1:
#         tape1.watch(x)  # x has to be tf tensor
#         # dy_dx = tape1.gradient(f(x), x)
#         with tf.GradientTape() as tape2:
#             tape2.watch(x)
#             f = f(x)
#             dy_dx = tape1.gradient(f, x)
#         d2y_dx2 = tape2.gradient(dy_dx, x)
#     return d2y_dx2 + 2

# my_diff_fns = [my_diff_fn]

# # Example trainig setup
# domain = (0, 2)
# i_max = 100
# layers = [100]
# hidden_layers = len(layers)
# s = Solver(i_max, domain, my_diff_fns, boundary_terms, xbs, hidden_layers, layers, sigmoid)
# s.compile_model('adam')
# epochs = 1000
# s.train_model(epochs=epochs)

# x_test = np.linspace(0, 2, 20)
# # x_tensor = tf.reshape(tf.convert_to_tensor(x), (20, 1))
# y = s.predict(x_test)

# x = np.linspace(0, 2, 1000)
# y_correct = - x**2 + 2 * x



# Example function 2 (???)
def boundary_term1(xb, fs):
    ''' f1(0) = 0 '''
    return fs[0](xb)

xb1 = 0

def boundary_term2(xb, fs):
    ''' f2(2) = 1 '''
    return fs[1](xb) - 1

xb2 = 2

def my_diff_fn1(x, fs):
    ''' f1'(x) - cos(x) - f1^2 - f2 + 1 + x^2 + sin^2(x) = 0 '''
    f1 = fs[0]
    f2 = fs[1]
    with tf.GradientTape() as tape:
        tape.watch(x)
    df1_dx = tape.gradient(f1, x)
    return df1_dx - tf.math.cos(x) - f1(x)**2 - f2(x) + 1 + x**2 + tf.math.sin(x)**2

def my_diff_fn2(x, fs):
    f1 = fs[0]
    f2 = fs[1]
    with tf.GradientTape() as tape:
        tape.watch(x)
    df2_dx = tape.gradient(f2, x)
    return df2_dx - 2 * x + (1 + x**2) * tf.math.sin(x) - f1(x) * f2(x)

boundary_terms = (boundary_term1, boundary_term2)
xbs = [xb1, xb2]
my_diff_fns = [my_diff_fn1, my_diff_fn2]


domain = (0, 2)
i_max = 100
layers = [100]
hidden_layers = len(layers)
s = Solver(i_max, domain, my_diff_fns, boundary_terms, xbs, hidden_layers, layers, sigmoid)
s.compile_model('adam')
epochs = 1000
# s.train_model(epochs=epochs)
# x = np.linspace(domain[0], domain[1], i_max)
# x = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), (i_max, 1))
# my_diff_fn1(x=x, fs=s.fs)


# import matplotlib.pyplot as plt


# plt.plot(x_test, y, label='NN')
# plt.plot(x, y_correct, label='Correct solution')
# plt.grid(True)
# plt.legend()
# plt.show()
