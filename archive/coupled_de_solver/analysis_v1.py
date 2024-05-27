'''
Analyse coupled_de_solver_v3.py with different parameters.
Especially study how the model learns the equations and how much time the training take.
'''

import coupled_de_solver_v3 as solver
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os


def plotting(predictions, eq1_diff_losses, correct, x_test, path, model_name):
    '''
    Creates the basic plot which compares
    - The correct and iterated solutions
    - Difference between the correct and iterated solutions
    - Differential loss of each iteration
    '''
    y1_correct = correct[0]
    y2_correct = correct[1]

    # Do the legend and the spacing of the plots better, especially more clear

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [5, 2, 2]})

    ax1.plot(x_test, y1_correct, label=r'$\phi_1$', color='red')
    ax1.plot(x_test, y2_correct, label=r'$\phi_2$', color='green')
    for i, prediction in enumerate(predictions):
        iter_i_y1 = prediction[:, 0]
        iter_i_y2 = prediction[:, 1]
        label1 = r'$\hat{\phi_1}$' + 'iter' + str(i + 1)
        label2 = r'$\hat{\phi_2}$' + 'iter' + str(i + 1)
        ax1.plot(x_test, iter_i_y1, label=label1, linestyle='dashed', color='red', linewidth=(i + 1) * 0.5)
        ax1.plot(x_test, iter_i_y2, label=label2, linestyle='dashed', color='green', linewidth=(i + 1) * 0.5)
    ax1.grid(True)
    ax1.legend(prop={'size': 8})
    ax1.set_ylabel(r'$\hat{\phi}$')

    for i, prediction in enumerate(predictions):
        iter_i_y1 = prediction[:, 0]
        # iter_i_y2 = prediction[:, 1]
        delta_1 = y1_correct - iter_i_y1
        # delta_2 = y2_correct - iter_i_y2
        label1 = r'$\Delta \phi_1$' + str(i) + '. iter'
        ax2.plot(x_test, delta_1, label=label1)
    ax2.grid(True)
    ax2.legend(prop={'size': 8})
    ax2.set_ylabel(r'$\phi_1 - \hat{\phi_1}$')

    for i, eq1_diff_loss in enumerate(eq1_diff_losses):
        label = r'$\hat{\phi_1}$ diff loss'
        ax3.plot(x_test, eq1_diff_loss, label=label)
    ax3.grid(True)
    ax3.legend(prop={'size': 8})
    ax3.set_ylabel(r'$\hat{F_1}$')

    plt.xlabel(r'$x$')
    fig.suptitle(model_name)

    file_name = path + 'a_plot'
    plt.savefig(fname=file_name)


def analyse_checkpoints(a, b, i_max, x_b, hlayers, num_of_checkpoints, x, path):
    '''
    Gives out the predictions of the model in checkpoint_1, checkpoint_2, ... or, in other words, the predictions for 1. iteration, 2. iteration, 
    '''
    checkpoint_path = './' + path + 'checkpoint'
    predictions = []
    eq1_diff_losses = []
    for i in range(1, num_of_checkpoints + 1):
        test_model = solver.Model(a, b, i_max, x_b, hlayers)
        test_model.build_model()
        test_model.compile_model()
        model_path = checkpoint_path + '_' + str(i)
        test_model.neural_net.load_weights(model_path)

        prediction = test_model.neural_net.predict(x)
        predictions.append(prediction)

        eq1_diff_loss = test_model.eq1(x)
        eq1_diff_losses.append(eq1_diff_loss)

    return predictions, eq1_diff_losses


def running_the_model(a, b, i_max, x_b, hlayers, epochs, domain_slicing, path, x_test_tensor):
    model = solver.Model(a, b, i_max, x_b, hlayers)
    model.build_model()
    model.compile_model()
    batch_size = i_max  # You can also try varying this
    checkpoint_path = path + 'checkpoint'
    predictions, eq1_diff_losses = model.train_model(epochs=epochs, batch_size=batch_size, domain_slicing=domain_slicing, checkpoint_path=checkpoint_path, x_test_tensor=x_test_tensor)
    return predictions, eq1_diff_losses

# Parameters
a = 0
b = 3
x_b = 0
i_max = 100


# Some test parameters to iterate over
# i_max_iterable = [10, 100]  # ????
hlayers_iterable = [[100], [500], [100, 100]]
epochs_iterable = [100, 500, 1000, 2000]
domain_slicing_iterable = [1, 2, 3, 4, 5]

x_test = np.linspace(a, b, i_max, dtype=np.float32)
y1_correct = -np.sqrt(6) * np.sin(np.sqrt(6) * x_test)
y2_correct = 2 * np.cos(np.sqrt(6) * x_test)
correct = (y1_correct, y2_correct)
x_test_tensor = tf.reshape(tf.convert_to_tensor(x_test), (i_max, 1))


for hlayers in hlayers_iterable:
    for epochs in epochs_iterable:
        for domain_slicing in domain_slicing_iterable:
            model_name = 'hlayers=' + str(hlayers) + '_epochs=' + str(epochs) + '_domain_slicing=' + str(domain_slicing)
            time_now = time.asctime()
            # time_now = int(time.time())
            path = 'saves/' + str(time_now) + '_' + model_name + '/'
            t0 = time.time()
            predictions, eq1_diff_losses = running_the_model(a, b, i_max, x_b, hlayers, epochs, domain_slicing, path, x_test_tensor)
            training_time = str(time.time() - t0)
            # predictions, eq1_diff_losses = analyse_checkpoints(a, b, i_max, x_b, hlayers, domain_slicing, x_test_tensor, path)
            if not os.path.isdir(path):
                os.makedirs(path)
            plotting(predictions, eq1_diff_losses, correct, x_test, path, model_name)
            info_file_name = path + '/info.txt'
            parameter_info = 'a='+str(a)+', b='+str(b)+', i_max='+str(i_max)+', hlayers='+str(hlayers)+', epochs='+str(epochs)+', domain_slicing='+str(domain_slicing)
            with open(info_file_name, 'w') as file:
                file.write('Model parameters: ' + parameter_info + '\n')
                file.write('Training time (seconds): ' + training_time)
