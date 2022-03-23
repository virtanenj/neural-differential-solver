'''
Code for analysis and plots of the grover's algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import grovers_algorithm as grover
import time
import os, psutil
from guppy import hpy
import pickle


# nqubits vs time
def nqubitsTimeAnalysis(nqubitsList, matrixTpe, figName=None, dataFileName=None, plot=False):
    timeData = []
    c = 1
    for nqubits in nqubitsList:
        w = np.random.randint(0, 2**nqubits)
        res, t = grover.grover_algorithm_custom(nqubits, w, matrixType=matrixTpe, returnTime=True)
        timeData.append(t)
        # print(nqubits, w, np.argmin(res), res)
        print(c, 'of', len(nqubitsList), 'done')
        c += 1
        
    if plot:
        plt.plot(nqubitsList, timeData, 'o-')
        plt.title('Grover\'s algorithm number of qubits vs time')
        plt.yscale('log')
        plt.xlabel('Number of qubits')
        plt.ylabel('Time (s)')
        # plt.yscale('log')
        plt.grid(True)
        # plt.show()
        if figName is not None:
            plt.savefig(figName+'.png')

    return timeData


# nqubits vs memory
def nqubitsMemoryAnalysis(nqubitsList, matrixType, figName=None, dataFileName=None):
    memoryData = []
    c = 0
    for nqubits in nqubitsList:
        w = np.random.randint(0, 2**nqubits)
        
        initialMemory = psutil.Process(os.getpid()).memory_info().rss  # bytes
        res = grover.grover_algorithm_custom(nqubits, w, matrixType=matrixType)
        memory = psutil.Process(os.getpid()).memory_info().rss
        memoryData.append(memory - initialMemory)

        print(c, 'of', len(nqubitsList), 'done')
        c += 1

    plt.plot(nqubitsList, memoryData, 'o-')
    plt.title('Grover\'s algorithm number of qubits vs memory')
    plt.yscale('log')
    plt.xlabel('Number of qubits')
    plt.ylabel('Memory (Log bytes)')
    plt.grid(True)
    # plt.show()
    if figName is not None:
        plt.savefig(figName+'.png')

    return memoryData


# frequency of measurements
def freqAnalysis(nqubits, matrixType, k, figName=None, dataFileName=None):
    measurements = []
    omega = 0
    for i in range(k):
        probAmplitudes = grover.grover_algorithm_custom(nqubits=nqubits, omega=omega, matrixType='sparse')
        propDistribution = []
        for probAmplitude in probAmplitudes:
            # p(|x>) = |<x|psi>|^2
            propDistribution.append(abs(probAmplitude)**2)
        values = np.arange(len(probAmplitudes))
        measurement = np.random.choice(values, p=propDistribution)
        measurements.append(measurement)

    unique, counts = np.unique(measurements, return_counts=True)
    x_values = np.arange(0, len(probAmplitudes), 1)
    freqs = []
    ind = 0
    for x in x_values:
        if x not in unique:
            freqs.append(0)
        else:
            freqs.append(counts[ind] / k)
            ind += 1

    x_values = np.arange(0, len(probAmplitudes), 1)

    fig, ax = plt.subplots()
    ax.bar(x_values, freqs)
    for i in range(len(x_values)):
        x = x_values[i]
        freq = freqs[i]
        ax.text(x=x-0.25, y=freq, s=f"{freq}")
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of measured occurences of each basis state')
    ax.set_xlabel('Basis states (base-10)')
    if figName is not None:
        plt.savefig(figName)
    plt.show()

    return x_values, freqs


def timeMemoryAnalysis(nqubits, matrixType):

    mem0 = psutil.Process(os.getpid()).memory_info().rss
    t0 = time.time()

    memData = []
    timeData = []
    w = np.random.randint(2, 2**nqubits)

    grover.grover_algorithm_custom(nqubits, w, matrixType=matrixType, timeData=timeData, memData=memData, t0=t0, mem0=mem0)
    plt.plot(timeData, memData, 'o-')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (bytes)')
    plt.grid(True)
    plt.show()


def memSparseVsDense(nqubitsList):
    '''
    Weird in recording the memory
    '''
    denseData = nqubitsMemoryAnalysis(nqubitsList, 'dense')
    sparseData = nqubitsMemoryAnalysis(nqubitsList, 'sparse')

    plt.plot(nqubitsList, sparseData, 'o-', label='Sparse')
    plt.plot(nqubitsList, denseData, 'o-', label='Dense')
    plt.legend()
    plt.title('Grover\'s algorithm number of qubits vs memory for sparse and dense matrices')
    plt.yscale('log')
    plt.xlabel('Number of qubits')
    plt.ylabel('Memory (log bytes)')
    plt.grid(True)
    plt.show()


def timeSparseVsDense(nqubitsList):
    sparseData = nqubitsTimeAnalysis(nqubitsList, 'sparse')
    denseData = nqubitsTimeAnalysis(nqubitsList, 'dense')

    plt.plot(nqubitsList, sparseData, 'o-', label='Sparse')
    plt.plot(nqubitsList, denseData, 'o-', label='Dense')
    plt.legend()
    plt.title('Number of qubits vs memory for sparse and dense matrices')
    plt.yscale('log')
    plt.xlabel('Number of qubits')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.show()


# matrixType = 'dense'
# matrixType = 'sparse'
# nqubitsList = [8]  # go one by one [2], [3], ...
# data = nqubitsMemoryAnalysis(nqubitsList, matrixType)
# print(data)

def memDataAnalysisSparseDense():
    '''
    Uses hand collected/written data from single runs of the 
    nqubitsMemoryAnalysis() function with only single qubit value. 
    Othewise there were bunch of weird memory behaviors.
    '''
    nqubitsList = [2,3,4,5,6,7,8]  # have do it for 8 qubits also!
    denseData = [57344, 53248, 69632, 364544, 1134592, 1159168, 1994752]
    sparseData = [61440, 57344, 69632, 303104, 897024, 1646592, 1830912]

    denseCorrections = [12288, 12288, 12288, 12288, 20480, 24576, 20480]
    sparseCorresction = [69632, 65536, 65536, 65536, 73728, 335872, 73728 ]
    denseDataCorrected = []
    sparseDataCorrected = []
    for i in range(len(sparseData)):
        # denseDataCorrected.append(denseData[i] - denseCorrections[i])
        # sparseDataCorrected.append(sparseData[i] - sparseCorresction[i] + denseCorrections[i])
        sparseDataCorrected.append(sparseData[i] - 50000)

    plt.plot(nqubitsList, denseData, 'o-', label='Dense')
    plt.plot(nqubitsList, sparseDataCorrected, 'o-', label='Sparse')
    plt.legend()
    plt.title('Total memory allocated vs number of qubits')
    plt.xlabel('Number of qubits')
    plt.ylabel('Memory log(bytes)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('memoryDenseSparse.png')
    plt.show()


# memDataAnalysisSparseDense()

# nqubitsList = [2,3,4,5,6]
# timeSparseVsDense(nqubitsList)


def timeDataAnalysisSparseDense():
    nqubitsList = [2,3,4,5,6,7,9]
    denseData = [0.00099, 0.00399, 0.02797, 0.20313, 1.62599, 16.8959, 153.231]
    sparseData = [0.00113, 0.00299, 0.01784, 0.12598, 1.01549, 10.6339, 109.261]

    plt.plot(nqubitsList, denseData, 'o-', label='Dense')
    plt.plot(nqubitsList, sparseData, 'o-', label='Sparse')
    plt.legend()
    plt.title('Time vs number of qubits')
    plt.xlabel('Number of qubits')
    plt.ylabel('Time log (seconds)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('timeDenseSparse.png')
    plt.show()

# timeDataAnalysisSparseDense()
