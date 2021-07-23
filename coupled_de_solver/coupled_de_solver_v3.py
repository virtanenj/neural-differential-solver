'''
Solves the coupled differential equations
    f'(x) + 3 * g(x) = 0,
    g'(x) - 2 * f(x) = 0
in the domain 0 <= x <= 3 (or more???) with the boundary conditions
    f(0) = 0,   g(0) = 2
'''


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model():
    def __init__(self, a, b, i_max, x_b, hlayers):
        self.i_max = i_max
        self.x = np.linspace(a, b, i_max, dtype=np.float32)
        self.x_tensor = tf.reshape(tf.convert_to_tensor(self.x), (self.i_max, 1))
        self.x_b = x_b
        self.x_b_tensor = tf.reshape(tf.convert_to_tensor(self.x_b), (1, 1))
        self.hlayers = hlayers
        self.neural_net = None
        # neural_net(x)[:, 0] == f1(x), neural_net(x)[:, 1] == f2(x), ...

    def build_model(self):
        input_layer = keras.layers.Input(shape=(1,)) # Two coupled equations
        hlayers_saved = []
        hlayer = keras.layers.Dense(self.hlayers[0], activation=tf.nn.sigmoid)(input_layer)
        hlayers_saved.append(hlayer)
        for i, hlayer in enumerate(self.hlayers[1:]):
            prev_layer = hlayers_saved[i - 1]
            hlayer = keras.layers.Dense(self.hlayers[i], activation=tf.nn.sigmoid)(prev_layer)
            hlayers_saved.append(hlayer)
        output_layer = keras.layers.Dense(2, activation=tf.identity)(hlayers_saved[-1])
        self.neural_net = keras.Model(inputs=input_layer, outputs=output_layer)

    @tf.function
    def f1(self, x):
        return tf.reshape(self.neural_net(x)[:, 0], (len(x), 1))

    @tf.function
    def f2(self, x):
        return tf.reshape(self.neural_net(x)[:, 1], (len(x), 1))

    def eq1(self, x):
        '''
        f'(x) + 3 * g(x) = 0
        '''
        with tf.GradientTape() as tape:
            tape.watch(x)
            y1 = self.f1(x)
            dy1_dx = tape.gradient(y1, x)
        return dy1_dx + 3 * self.f2(x)

    def eq2(self, x):
        '''
        g'(x) - 2 * f(x) = 0
        '''
        with tf.GradientTape() as tape:
            tape.watch(x)
            y2 = self.f2(x)
            dy2_dx = tape.gradient(y2, x)
        return dy2_dx - 2 * self.f1(x)

    def coupled_diff_eq_1_squarred(self, x):
        return self.eq1(x)**2

    def coupled_diff_eq_2_squarred(self, x):
        return self.eq2(x)**2

    def boundary_term_1(self):
        return self.f1(self.x_b_tensor)

    def boundary_term_2(self):
        return self.f2(self.x_b_tensor) - 2

    def my_loss_fn(self):
        def loss(y_true, y_pred):
            x_training = y_true
            sum_over_F1 = tf.math.reduce_sum(tf.map_fn(self.coupled_diff_eq_1_squarred, x_training))
            sum_over_F2 = tf.math.reduce_sum(tf.map_fn(self.coupled_diff_eq_2_squarred, x_training))
            sum_over_boundary_terms = self.boundary_term_1()**2 + self.boundary_term_2()**2
            return (sum_over_F1 + sum_over_F2) / self.i_max + sum_over_boundary_terms
        return loss

    def compile_model(self, optimizer='adam'):
        self.neural_net.compile(loss=self.my_loss_fn(), optimizer=optimizer)

    def train_model(self, epochs, batch_size, domain_slicing=1, checkpoint_path=None, x_test_tensor=None):
        predictions = []
        eq1_diff_losses = []
        if domain_slicing != 1:
            # Divide the domain into equal siced parts
            for i in range(1, domain_slicing + 1):
                if i == domain_slicing:
                    # Makes sure, that the last point is included in all cases
                    i_max_now = self.i_max
                else:
                    i_max_now = int(self.i_max / domain_slicing) * i
                x_tensor_now = self.x_tensor[:i_max_now]
                self.neural_net.fit(x=x_tensor_now, y=x_tensor_now, epochs=epochs, batch_size=batch_size)
                if checkpoint_path is not None:
                    path_to_this_checkpoint = checkpoint_path + '_' + str(i)  # NOTE: indexing starts from 1
                    self.neural_net.save_weights(path_to_this_checkpoint)
                if x_test_tensor is not None:
                    prediction = self.neural_net.predict(x_test_tensor)
                    eq1_diff_loss = self.eq1(x_test_tensor)
                    predictions.append(prediction)
                    eq1_diff_losses.append(eq1_diff_loss)
        else:
            # No slicing
            self.neural_net.fit(x=self.x_tensor, y=self.x_tensor, epochs=epochs, batch_size=batch_size)
            if x_test_tensor is not None:
                prediction = self.neural_net.predict(x_test_tensor)
                eq1_diff_loss = self.eq1(x_test_tensor)
                predictions.append(prediction)
                eq1_diff_losses.append(eq1_diff_loss)
        return predictions, eq1_diff_losses


# a = 0
# b = 3
# i_max = 100
# x_b = 0
# hlayers = [100]

# model = Model(a, b, i_max, x_b, hlayers)
# model.build_model()
# model.compile_model()

# epochs = 100  # epochs per iteration 
# batch_size = i_max
# domain_slicing = 3  # How the domain is sliced == how many iterations

# model.train_model(epochs, batch_size, domain_slicing=domain_slicing)

# x_test = np.linspace(a, b, 100, dtype=np.float32)
# x_test_tensor = tf.reshape(tf.convert_to_tensor(x_test), (i_max, 1))

# prediction = model.neural_net.predict(x_test_tensor)
# print(prediction[:, 0])
