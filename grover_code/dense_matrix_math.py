import numpy as np


def complex_multiplication(z1, z2):
    return complex(z1.real * z2.real - z1.imag * z2.imag, z1.imag * z2.real - + z1.real * z2.imag)


def vector_tensor_prod(v, w):
    result = []
    for vi in v:
        for wi in w:
            result.append(vi*wi)
    return np.array(result)


def matrix_tensor_prod(V, W):
    '''
    Also known as the Kronecker product
    '''
    nV = len(V)  # V matrix width
    mV = len(V[0])  # V matrix height
    nW = len(W)  # W matrix width
    mW = len(W[0])  # W matrix height
    result = np.zeros((nV * nW, mV * mW), dtype=complex)
    counter1 = 0
    for i in range(nV):
        counter2 = 0
        for j in range(mV):
            for k in range(nW):
                for l in range(mW):
                    # result[k + counter1, l + counter2] = V[i, j] * W[k, l]
                    result[k + counter1, l + counter2] = complex_multiplication(V[i, j], W[k, l])
            counter2 = (j + 1) * mW
        counter1 = (i + 1) * nV
    return result


def matrix_multiplication(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        raise Exception('Can not multiplie A and B because of their shape')
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return np.array(C)


def matrix_vector_prod_not(matrix, vector):
    if matrix.shape[1] != len(vector):
        raise Exception('matrix width must be the same as vector lenght')
    res = np.zeros(len(vector), dtype=complex)
    for i in range(len(matrix)):
        for j in range(len(vector)):
            pass
    return res


def matrix_vector_prod(matrix, vector):
    return matrix.dot(vector)

