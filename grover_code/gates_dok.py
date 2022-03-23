import numpy as np
import sparse_matrix_dok as sm


class QuantumGate:
    def __init__(self, name):
        '''
        name: the name of the gate as a string
        '''
        self.name = name
        self.matrix = sm.SparseMatrix(np.array([[]], dtype = complex))
        self.identity = sm.sparseIdentity(2)


class Hadamard(QuantumGate):
    def __init__(self):
        super().__init__('Hadamard')
        self.matrix = sm.SparseMatrix((1.0/2.0**0.5) * np.array([[1,1], [1,-1]], dtype = complex ))

class PauliX(QuantumGate):
    def __init__(self):
        super().__init__('Pauli X')
        self.matrix = sm.SparseMatrix(np.array([[0,1], [1,0]], dtype = complex ))

class ControlledV(QuantumGate):
    def __init__(self):
        super().__init__('Controlled V')
        self.matrix = sm.SparseMatrix((np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1j]], dtype=complex)))

class Cnot(QuantumGate):
    def __init__(self):
        super().__init__('Controlled Not')
        self.matrix = sm.SparseMatrix(np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype = complex))

class Rot(QuantumGate):
    def __init__(self, phi):
        super().__init__('Rotation')
        self.matrix = sm.SparseMatrix(np.array([[1, 0], [0, np.exp(1j * phi)]]))

class SGate(QuantumGate):
    def __init__(self):
        super().__init__('S Gate')
        self.matrix = sm.SparseMatrix(np.array([[1, 0], [0, 1j]]))

class Identity(QuantumGate):
    def __init__(self):
        super().__init__('Identity')
        self.matrix = sm.SparseMatrix(np.identity(2))

class Swap(QuantumGate):
    def __init__(self):
        super().__init__('Swap')
        self.matrix = sm.SparseMatrix(np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]], dtype = complex))


class Blank(QuantumGate):
    def __init__(self):
        super().__init__('Blank')
        self.matrix = sm.SparseMatrix(np.array([[1],[1]]))

class ControlledZ(QuantumGate):
    def __init__(self):
        super().__init__('Controlled Z')
        self.matrix = sm.SparseMatrix(np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, -1]]))

class ZGate(QuantumGate):
    def __init__(self):
        super().__init__('Z Gate')
        self.matrix = sm.SparseMatrix(np.array([[1, 0],[0, -1]]))

# Gates for Grover's algorithm
class Oracle(QuantumGate):
    def __init__(self, nqubits, omega):
        '''
        nqubits: number of qubits
        omega: the value we are searching
        '''
        super().__init__('Oracle')
        diag = np.ones(2**nqubits, dtype=complex)
        diag[omega] = -1
        my_sm = {}
        for i in range(2**nqubits):
            my_sm[(i,i)] = diag[i]
        self.matrix = sm.SparseMatrix(my_sm, rows=2**nqubits, columns=2**nqubits, sparse=True)


class DiffusionHelper(QuantumGate):
    '''
    The (Grover) diffusion operator: 2|psi><psi| - I = H^n (2|0><0| - I) H^n 
    Here the Diffusion helper: 2|0><0| - I = diag(1,-1,...,-1)
    '''
    def __init__(self, nqubits):
        super().__init__('Diffusion helper gate')
        diag = np.ones(2**nqubits)
        diag = diag * (-1)
        diag[0] = 1
        my_sm = {}
        for i in range(2**nqubits):
            my_sm[(i, i)] = diag[i]
        self.matrix = sm.SparseMatrix(matrix=my_sm, rows=2**nqubits, columns=2**nqubits, sparse=True)


class Diffusion(QuantumGate):
    def __init__(self, nqubits):
        super().__init__('Diffusion gate')
        H = Hadamard()
        HTransformation = H.matrix
        for _ in range(nqubits - 1):
            HTransformation = sm.tensorSparse(H.matrix, HTransformation)
        diff_helper_mat = DiffusionHelper(nqubits).matrix
        # Calculates H^n.diff_helper_mat.H^n = H^n.(2|0><0| - I).H^n
        my_mat = sm.multiplySparse(diff_helper_mat, HTransformation)
        my_mat = sm.multiplySparse(HTransformation, my_mat)
        self.matrix = my_mat
