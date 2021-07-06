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


# A better example with clear analytic solution
def coupled_diff_eq_1(x):
    '''
    f'(x) + 3 * g(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1 = f1(x)
        dy1_dx = tape.gradient(y1, x)
    return (dy1_dx + 3 * f2(x))**2


def coupled_diff_eq_2(x):
    '''
    g'(x) - 2 * f(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y2 = f2(x)
        dy2_dx = tape.gradient(y2, x)
    return (dy2_dx - 2 * f1(x))**2


# The boundaries are f(0) = 0, g(0) = 2. The domain is 0 <= x <= 3.


def my_loss_fn(x):
    def loss(y_true, y_pred):
        # L = (1/i_max)(F_1^2 + F_2^2) + (f_1(0) - 0)^2 + (f_2(0) - 1)^2
        sum_over_F1 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_1, x))
        sum_over_F2 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_2, x))
        sum_over_boundary_terms = f1(x_b)**2 + (f2(x_b) - 2)**2   # The boundaries
        return (sum_over_F1 + sum_over_F2) / i_max + sum_over_boundary_terms
    return loss


def build_model(units):
    input_layer = keras.layers.Input(shape=(1,))
    hidden_layer = keras.layers.Dense(units, activation=tf.nn.sigmoid)(input_layer)
    output_layer = keras.layers.Dense(2, activation=tf.identity)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model



# Do a for-loop to step-by-step increase the boundaries
# The model should retain its weights and biases.

# The point at which the bondary conditions are (must be global)
x_b = tf.reshape(tf.convert_to_tensor(0), (1, 1))

# Initially 
a = 0
# b_init = 1
b_final = 3
steps = 3
b_steps = np.linspace(a, b_final, steps)
epochs_per_step = 5000
i_max = 100
units = 10
x = np.linspace(a, b_steps[1], i_max, dtype=np.float32)
x_tensor = tf.reshape(tf.convert_to_tensor(x), (i_max, 1))

neural_net = build_model(units=units)
neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
print('b =', b_steps[1])
neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs_per_step)

checkpoints = []
checkpoints.append(neural_net.save_weights('checkpoints/checkpoint_1'))
for i in range(2, len(b_steps)):
    print('b =', b_steps[i])
    x = np.linspace(a, b_steps[i], i_max, dtype=np.float32)
    x_tensor = tf.reshape(tf.convert_to_tensor(x), (i_max, 1))
    neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs_per_step)
    checkpoints.append(neural_net.save_weights('checkpoints/checkpoint_' + str(i)))


x_test = np.linspace(a, b_steps[-1], 100)
y_predictions = neural_net.predict(x_test)

iteration_1_weights = checkpoints[0]
test_model = build_model(units=units)
test_model.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
test_model.load_weights('checkpoints/checkpoint_1')
it_1_predictions = test_model.predict(x_test)
y1_it_1 = it_1_predictions[:, 0]
y2_it_1 = it_1_predictions[:, 1]

y1_prediction = y_predictions[:, 0]
y2_prediction = y_predictions[:, 1]
y1_correct = -np.sqrt(6) * np.sin(np.sqrt(6) * x_test)
y2_correct = 2 * np.cos(np.sqrt(6) * x_test)


import matplotlib.pyplot as plt

plt.plot(x_test, y1_prediction, label='final model y1', marker='.')
plt.plot(x_test, y2_prediction, label='final model y2', marker='.')
plt.plot(x_test, y1_correct, label='correct solution for y1')
plt.plot(x_test, y2_correct, label='correct solution for y2')

plt.plot(x_test, y1_it_1, label='1. iteration y1', marker='^')
plt.plot(x_test, y2_it_1, label='1. iteration y2', marker='^')

plt.legend()
name = 'coupled_diff_example_with_' + str(steps) + '_steps,_' + str(epochs_per_step) + '_epochs_per_step,_' + str(i_max) +'_training_points,_' + str(units) + '_units'
complete_name = 'figures/' + name
# plt.savefig(fname=complete_name)
plt.show()
