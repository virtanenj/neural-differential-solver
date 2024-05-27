'''
Analyse coupled_de_solver_v3.py with different parameters.
Especially study how the model learns the equations and how much time the training take.

To do:
- Increasing the number of layers with other things constant
- Increasing epochs (for some config. of layers) with other things constant
- Demonstrate the need to use iterations in a large domain?
- Solutions for some other DEs

Note:
- How would this compare to TensorBoard? (ks. https://keras.io/api/callbacks/tensorboard/)
'''

import coupled_de_solver_v3 as solver
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os


def plotting(predictions, eq1_diff_losses, histories, correct, x_test, path, model_name, save_figs=True):
    '''
    Creates the basic plot which compares
    - The correct and iterated solutions
    - Difference between the correct and iterated solutions
    - Differential loss of each iteration

    To do (here): another plot for the loss across the training using the histories parameter (remember: how is the loss calculated??)
    '''
    y1_correct = correct[0]
    y2_correct = correct[1]

    # Plots along the domain of the equations
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})

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
    ax1.legend(prop={'size': 8}, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    ax1.set_ylabel(r'$\hat{\phi}$')

    for i, prediction in enumerate(predictions):
        iter_i_y1 = prediction[:, 0]
        # iter_i_y2 = prediction[:, 1]
        delta_1 = y1_correct - iter_i_y1
        # delta_2 = y2_correct - iter_i_y2
        label1 = r'$\Delta \phi_1$' + str(i+1) + '. iter'
        ax2.plot(x_test, delta_1, label=label1)
    ax2.grid(True)
    ax2.legend(prop={'size': 8}, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    ax2.set_ylabel(r'$\phi_1 - \hat{\phi_1}$')

    for i, eq1_diff_loss in enumerate(eq1_diff_losses):
        label = r'$\hat{F_1}$ ' + str(i+1) + '. iter'
        ax3.plot(x_test, eq1_diff_loss, label=label)
    ax3.grid(True)
    ax3.legend(prop={'size': 8}, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    ax3.set_ylabel(r'$\hat{F_1}$')

    plt.xlabel(r'$x$')
    fig.suptitle(model_name)
    fig.set_size_inches(10, 10)
    if save_figs:
        file_name_1 = path + 'domain_plot'
        plt.savefig(fname=file_name_1)

    # Plot of the loss along training
    plt.figure()
    for i, history in enumerate(histories):
        plt.plot(list(history.history.values())[0], label=str(i+1)+'. iter')
    # plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of each iteration: ' + model_name)
    plt.legend()
    plt.grid(True)
    if save_figs:
        file_name_2 = path + 'loss_plot'
        plt.savefig(fname=file_name_2)
    # plt.show()



def plot_and_save(predictions, eq1_diff_losses, histories, x_test, path, model_name, running_time, parameter_info, save_fig=True):
    if not os.path.isdir(path):
        os.makedirs(path)
    plotting(predictions, eq1_diff_losses, histories, correct, x_test, path, model_name, save_fig)
    info_file_name = path + '/info.txt'
    with open(info_file_name, 'w') as file:
        file.write('Model parameterhistorys: ' + parameter_info + '\n')
        file.write('Training time (seconds): ' + running_time)


def running_the_model(a, b, i_max, x_b, hlayers, epochs, domain_slicing, path, x_test_tensor):
    model = solver.Model(a, b, i_max, x_b, hlayers)
    model.build_model()
    model.compile_model()
    batch_size = i_max  # You can also try varying this
    checkpoint_path = path + 'checkpoint'
    return model.train_model(epochs=epochs, batch_size=batch_size, domain_slicing=domain_slicing, checkpoint_path=checkpoint_path, x_test_tensor=x_test_tensor)


# Parameters
a = 0
b = 3
x_b = 0
i_max = 1000

# Some test parameters to iterate over
# i_max_iterable = [10, 100]  # ????
hlayers_iterable = [[100,100,100]]
epochs_iterable = [1000]
domain_slicing_iterable = [3]

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
            predictions, eq1_diff_losses, histories = running_the_model(a, b, i_max, x_b, hlayers, epochs, domain_slicing, path, x_test_tensor)
            running_time = str(time.time() - t0)
            parameter_info = 'a='+str(a)+', b='+str(b)+', i_max='+str(i_max)+', hlayers='+str(hlayers)+', epochs='+str(epochs)+', domain_slicing='+str(domain_slicing)
            plot_and_save(predictions, eq1_diff_losses, histories, x_test, path, model_name, running_time, parameter_info, save_fig=True)
