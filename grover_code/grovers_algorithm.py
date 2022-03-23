'''
Grover's algorithm:
    -- State preparation -- Oracle -- Diffusion -- Oracle -- Diffusion -- ... -- Measurement
where -- Oracle -- Diffusion -- is applied ceil(pi sqrt(nqubits)  /4) times.
Note that an ancillary qubit is added which Oracle acts on while Diffusion does not.
'''

import numpy as np
import gates_dok as g
# from combined_circuit import QuantumCircuit
import sparse_matrix_dok as sm
import dense_matrix_math as dmm

import psutil, os
import time


def grover_algorithm_custom(nqubits, omega, matrixType='sparse', timeData=None, memData=None, t0=None, mem0=None, returnTime=False):
    '''
    nqubits: number of qubits (base-10)
    omega: qubit value we are searhing for (base-10)
    matrixType: 'sparse' or 'dense'

    This function builds grover's algorithm directly using (sparse) matrices and 
    doing related tensor products and matrix multiplications (and gives correct 
    results without errors).
    '''
    t1 = time.time()

    # For using wither sparse or dense matrices
    if matrixType == 'sparse':
        H = g.Hadamard().matrix
        O = g.Oracle(nqubits, omega).matrix
        Diff = g.Diffusion(nqubits).matrix
    elif matrixType == 'dense':
        H = g.Hadamard().matrix.toNPArray()
        O = g.Oracle(nqubits, omega).matrix.toNPArray()
        Diff = g.Diffusion(nqubits).matrix.toNPArray()
    else:
        raise Exception('matrixType should by either \'sparse\' or \'dense\'')
    
    qubit0 = np.array([1,0])
    qubit0n = qubit0
    for i in range(nqubits - 1):
        qubit0n = dmm.vector_tensor_prod(qubit0n, qubit0)

    # 1) Prepare states 
    my_mat = H
    for _ in range(nqubits - 1):
        my_mat = matrixTensorProd(H, my_mat, matrixType)

    # 2) Apply grover/diffusion gate ceil(pi sqrt(nquits) / 4) times
    iterations = int(np.ceil(np.pi * np.sqrt(nqubits) / 4))
    
    # For some weird behaviour this produces correct result for 2 qubits
    # instead of having 2 iterations
    if nqubits == 2:
        iterations = 1

    for i in range(iterations):
        # Calculates Diff.O = H^n.(2|0><0| - I).H^n.O
        my_mat = multiplyMat(O, my_mat, matrixType)
        my_mat = multiplyMat(Diff, my_mat, matrixType)
    
    t2 = time.time()
    if matrixType == 'sparse':
        final_res = my_mat.toNPArray() @ qubit0n  # <--- time and memory penalty
        # final_res = sm.multiplySparseVec(my_mat, qubit0n)
        if returnTime:
            return final_res, t2 - t1
        return final_res
    elif matrixType == 'dense':
        final_res = my_mat.dot(qubit0n)
        if returnTime:
            return final_res, t2 - t1
        return final_res


def matrixTensorProd(firstMat, secondMat, matrixType):
    if matrixType == 'sparse':
        return sm.tensorSparse(firstMat, secondMat)
    elif matrixType == 'dense':
        return dmm.matrix_tensor_prod(firstMat, secondMat)
    else:
        raise Exception('matrixType should by either \'sparse\' or \'dense\'')


def multiplyMat(firstMat, secondMat, matrixType):
    if matrixType == 'sparse':
        return sm.multiplySparse(firstMat, secondMat)
    elif matrixType == 'dense':
        # return firstMat @ secondMat  # <---
        return dmm.matrix_multiplication(firstMat, secondMat)
    else:
        raise Exception('matrixType should by either \'sparse\' or \'dense\'')


def recTimeMem(timeData, memData, t0, mem0):
    mem = psutil.Process(os.getpid()).memory_info().rss
    memData.append(mem - mem0)
    t = time.time()
    timeData.append(t - t0)


if __name__=='__main__':
    nqubits = 3
    w = 0
    res, t = grover_algorithm_custom(nqubits, w, matrixType='sparse')
 
    print('----Results----')
    print(res)
    print(np.argmax(res), max(res), len(res))
