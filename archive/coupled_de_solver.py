import numpy as np
# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3
import tensorflow as tf
from tensorflow import keras

# tf.keras.backend.set_floatx('float64')  # ??


# Is this necessary
@tf.function
def f1(x):
    return tf.reshape(neural_net(x)[:, 0], (len(x), 1))


@tf.function
def f2(x):
    return tf.reshape(neural_net(x)[:, 1], (len(x), 1))


# Do this better...
# Example of coupled differential equations

def coupled_diff_eq_1(x):
    '''
    f'(x) + 3 * g(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1 = f1(x)
        dy1_dx = tape.gradient(y1, x)
    return dy1_dx + 3 * f2(x)

def coupled_diff_eq_2(x):
    '''
    g'(x) - 2 * f(x) = 0
    '''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y2 = f2(x)
        dy2_dx = tape.gradient(y2, x)
    return dy2_dx - 2 * f1(x)


def coupled_diff_eq_1_sqrt(x):
    return coupled_diff_eq_1(x)**2


def coupled_diff_eq_2_sqrt(x):
    return coupled_diff_eq_2(x)**2

# The domain is 0 <= x <= 3.
# The boundaries are f(0) = 0, g(0) = 2.


def my_loss_fn(x):
    def loss(y_true, y_pred):
        # L = (1/i_max)(F_1^2 + F_2^2) + (f_1(0) - 0)^2 + (f_2(0) - 1)^2
        # coupled_diff_eq_1_squared = coupled_diff_eq_1(x)**2
        # coupled_diff_eq_2_squared = coupled_diff_eq_2(x)**2
        sum_over_F1 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_1_sqrt, x))
        sum_over_F2 = tf.math.reduce_sum(tf.map_fn(coupled_diff_eq_2_sqrt, x))
        sum_over_boundary_terms = f1(x_b)**2 + (f2(x_b) - 2)**2
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

# Hyperparameters
a = 0
b_final = 3
steps = 3  # Steps of increasing the boundry b
b_steps = np.linspace(a, b_final, steps + 1)[1:]  # Do not include b=0
epochs_per_step = 50
i_max = 100
# Should the number of training points be incresed each step?
x = np.linspace(a, b_steps[0], i_max, dtype=np.float32)
x_tensor = tf.reshape(tf.convert_to_tensor(x), (i_max, 1))
units = 10


# Building and compiling the model
neural_net = build_model(units=units)
neural_net.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
# Training and saving each iteration as a checkpoint
checkpoints = []
neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs_per_step)
checkpoints.append(neural_net.save_weights('checkpoints/checkpoint_0'))

# The domain for testing the model (????)
x_test = np.linspace(a, b_steps[-1], 100, dtype=np.float32)
x_test_tensor = tf.reshape(tf.convert_to_tensor(x_test), (i_max, 1))
eq1_diff_losses = []  # Saving the differential loss at each iteration in the whole domain

for i in range(1, len(b_steps)):
    x = np.linspace(a, b_steps[i], i_max, dtype=np.float32)
    x_tensor = tf.reshape(tf.convert_to_tensor(x), (i_max, 1))
    neural_net.fit(x=x_tensor, y=x_tensor, batch_size=i_max, epochs=epochs_per_step)
    checkpoints.append(neural_net.save_weights('checkpoints/checkpoint_' + str(i)))
    # Differential loss
    eq1_diff_loss = coupled_diff_eq_1(x_test_tensor)
    eq1_diff_losses.append(eq1_diff_loss)

# Testing the model for each checkpoint
neural_net.predict(x_test)

y1_correct = -np.sqrt(6) * np.sin(np.sqrt(6) * x_test)
y2_correct = 2 * np.cos(np.sqrt(6) * x_test)

predictions = []
for i in range(steps):
    test_model = build_model(units=units)
    test_model.compile(loss=my_loss_fn(x_tensor), optimizer='adam')
    file_path_i = 'checkpoints/checkpoint_' + str(i + 1)
    test_model.load_weights(file_path_i)
    iter_i_predictions = test_model.predict(x_test)
    predictions.append(iter_i_predictions)


# # Plotting the correct solution and solutions for each iteration
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [5, 2, 2]})

ax1.plot(x_test, y1_correct, label=r'$\phi_1$', color='red')
ax1.plot(x_test, y2_correct, label=r'$\phi_2$', color='green')


# plt.plot(x_test, y1_correct, label='correct solution for y1')
# plt.plot(x_test, y2_correct, label='correct solution for y2')

for i, prediction in enumerate(predictions):
    iter_i_y1 = prediction[:, 0]
    iter_i_y2 = prediction[:, 1]
    label1 = r'$\hat{\phi_1}$' + 'iter' + str(i + 1)
    label2 = r'$\hat{\phi_2}$' + 'iter' + str(i + 1)
    ax1.plot(x_test, iter_i_y1, label=label1, linestyle='dashed', color='red', linewidth=(i+1)*0.5)
    ax1.plot(x_test, iter_i_y2, label=label2, linestyle='dashed', color='green', linewidth=(i+1)*0.5)

ax1.grid(True)
ax1.legend(prop={'size': 8})
ax1.set_ylabel(r'$\hat{\phi}$')

# Plot y_correct - y_iteration
for i, prediction in enumerate(predictions):
    iter_i_y1 = prediction[:, 0]
    # iter_i_y2 = prediction[:, 1]
    delta_1 = y1_correct - iter_i_y1
    # delta_2 = y2_correct - iter_i_y2
    label1 = r'$\Delta \phi_1$' + str(i + 1) + '. iter'
    ax2.plot(x_test, delta_1, label=label1)
ax2.grid(True)
ax2.legend(prop={'size': 8})
ax2.set_ylabel(r'$\phi_1 - \hat{\phi_1}$')

# Plot differential lost
for i, eq1_diff_loss in enumerate(eq1_diff_losses):
    label = r'$\hat{\phi_1}$ diff loss'
    ax3.plot(x_test, eq1_diff_loss, label=label)
ax3.grid(True)
ax2.legend(prop={'size': 8})
ax3.set_ylabel(r'$\hat{F_1}$')


plt.xlabel(r'$x$')

# name = 'coupled_diff_example_with_' + str(steps) + '_steps,_' + str(
#     epochs_per_step) + '_epochs_per_step,_' + str(i_max) +'_training_points,_' + str(units) + '_units'
# file_name = 'figures/' + name
# plt.savefig(fname='new_fig')
plt.show()

# To do: y_correct - y_{iteration-1}
# To do: plot of the differential contribution F across the entire domain
