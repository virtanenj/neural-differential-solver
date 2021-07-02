'''
More general model that should also be more simple to use
'''

import numpy as np
# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3
import tensorflow as tf
from tensorflow import keras


class DE_solver():
    def __init__(self, x, i_max, differential_equation=None, boundary_conditions=None):
        self.x = x  # How would you consider multidimensional inputs, f(x1, x2, ...)
        self.i_max = i_max  # number of discrete x
        self.x_tensor = tf.reshape(tf.convert_to_tensor(self.x), (self.i_max, 1))
        self.model = None  # the solution(s)

        # How to do these???
        self.differential_equation = differential_equation  # F_m(...) = 0
        self.boundary_conditions = boundary_conditions  # list of boundry conditions in the form of grad^p(f_m(x_b)) - K(x_b) = 0

    def build_model(self, layer_units, hidden_layers):
        if type(layer_units) == int:
            layer_units = [layer_units]

        if len(layer_units) != hidden_layers:
            raise Exception('The lenght of the layer_units does not match hidden_layers')

        input_layer = keras.layers.Input(shape=(1,))
        hidden_layer = keras.layers.Dense(layer_units[0], activation=tf.nn.sigmoid)(input_layer)
        # This is for multiple hidden layers
        for i in range(1, hidden_layers):
            prev_hidden_layer = hidden_layer
            hidden_layer = keras.layers.Dense(layer_units[i], activation=tf.nn.sigmoid)(prev_hidden_layer)
        output_layer = keras.layers.Dense(1, activation=tf.identity)(hidden_layer)
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)

    # ???????
    def this_differential_equation(self, x):
        return self.differential_equation(x=x, f=self.model)
    
    def this_boundary_conditions(self):
        return self.boundary_conditions(f=self.model)

    # How to input these or similar equations into the class????
    def test_equation(self, x):
        '''
        d^2f/dx^2 + 2 = 0

        Note: x must be tensor
        '''
        with tf.GradientTape() as tape_1:
            tape_1.watch(x)
            with tf.GradientTape() as tape_2:
                tape_2.watch(x)
                y = self.model(x)
                dy_dx = tape_2.gradient(y, x)
            d2y_dx2 = tape_1.gradient(dy_dx, x)

        return (d2y_dx2 + 2)**2

    def test_boundary_conditions(self):
        '''
        f(0) = 0 <=> f(0) - 0 = 0,
        f(2) = 0 <=> f(2) - 0 = 0
        
        Note: should the boundary values of x be hardcoded?
        '''
        xb1 = tf.reshape(tf.convert_to_tensor(0), (1, 1))
        xb2 = tf.reshape(tf.convert_to_tensor(2), (1, 1))
        return [(self.model(xb1) - 0)**2, (self.model(xb2) - 0)**2]


    def my_loss_fn(self, x):
        # Is x necessary to have as parameter if it is already an instance variable?
        def loss(y_true, y_pred):
            '''
            Loss = 1/i_max sum_{i,m} F_m(x_i, f_m(x_i), grad(f_m(x_i)), ...)^2 
            + sum_{B.C.} (grad^p(f_m(x_b)) - K(x_b))^2
            '''
            # Should the squaring and summation be done here or elsewhere?
            sum_over_F = tf.math.reduce_sum(tf.map_fn(self.test_equation, x))
            # sum_over_boundary_terms = sum([bc**2 for bc in self.test_boundary_conditions()])
            sum_over_boundary_terms = sum(self.test_boundary_conditions())
            return sum_over_F / self.i_max + sum_over_boundary_terms
        return loss

    def compile_model(self):
        self.model.compile(loss=self.my_loss_fn(self.x_tensor), optimizer='adam')

    def train(self, epochs):
        self.model.fit(x=self.x_tensor, y=self.x_tensor, batch_size=self.i_max, epochs=epochs)


# def test_equation(x, f):
#     '''
#     d^2f/dx^2 + 2 = 0

#     Note: x must be tensor
#     '''
#     with tf.GradientTape() as tape_1:
#         tape_1.watch(x)
#         with tf.GradientTape() as tape_2:
#             tape_2.watch(x)
#             y = f(x)
#             dy_dx = tape_2.gradient(y, x)
#         d2y_dx2 = tape_1.gradient(dy_dx, x)

#     return (d2y_dx2 + 2)**2


# def test_boundary_conditions(f):
#     '''
#     f(0) = 0 <=> f(0) - 0 = 0,
#     f(2) = 0 <=> f(2) - 0 = 0
    
#     Note: should the boundary values of x be hardcoded?
#     '''
#     return [f(0) - 0, f(2) - 0]


a = 0
b = 2
i_max = 100
x = np.linspace(a, b, i_max, dtype=np.float32)

test_solver = DE_solver(x, i_max)
test_solver.build_model(layer_units=[100], hidden_layers=1)
# test_solver.differential_eqs = test_equation
# test_solver.boundaries = test_boundary_conditions
test_solver.compile_model()
test_solver.train(epochs=1000)


x_test = np.linspace(a, b, 20)
y_prediction = test_solver.model.predict(x_test)

x_correct = np.linspace(a, b, 100)
y_correct = -x_correct**2 + 2 * x_correct

import matplotlib.pyplot as plt

plt.plot(x_test, y_prediction, label='The model', marker='o')
plt.plot(x_correct, y_correct, label='Correct solution')
plt.legend()
plt.show()
