import numpy as np
import gates as g
import sparse_matrix as sm


identity = np.array([[1, 0], [0, 1]])
class QuantumCircuit:
    def __init__(self, nqubits):
        '''
        nqubits: number of qubits
        '''
        self.nqubits = nqubits
        self.gates = []
        self.size = 0
        self.matrix = None  # nqubits^2 x nqubits^2
        self.positions = [[g.Identity()]*nqubits]
        self.currentPosition = 0
        self.currentBitsUsed = {i:False for i in range(self.nqubits)}

    def reset(self):
        '''
        reset the quantum circuit to an empty version of itself
        '''
        self.gates = []
        self.size = 0
        self.matrix = None  # nqubits^2 x nqubits^2
        self.positions = [[g.Identity()] * self.nqubits]
        self.currentPosition = 0
        self.currentBitsUsed = {i:False for i in range(self.nqubits)}

    def moveForward(self):
        '''
        advance the pointer for the current set of gates
        '''
        self.currentPosition += 1
        self.positions.append([g.Identity()]*self.nqubits)
        self.currentBitsUsed = {i:False for i in range(self.nqubits)}

    def add_gate(self, gate, bits, advance=False):
        '''
        gate: the gate to apply
        bits: the bits the gate applies to (list of integrers!)
        advance: default to false; used for the recursive call when adding swaps to ensure they are not overwritten
        
        Note: 
        - gates must be added in order from left to right but not necessarily in vertical order
        For example consider circuit with the 2 qubit CNOT gate and some other arbitrary 1 qubit gates A, B, ...
        ---A---.---
               |
        ---C---+---

        -------E---
        This could be implented by:

        add_gate(A, 0)
        add_gate(C, 1)
        add_gate(CNOT, 0, 1)
        add_gate(E, 2)

        But could equally correctly be implemented by:

        add_gate(C, 1)
        add_gate(A, 0)
        add_gate(E, 2)
        add_gate(CNOT, 0, 1)
        '''
        # Takes care of some errors
        if type(bits[0]) is list:
            bits = bits[0]
        
        for i in list(bits):
            if self.currentBitsUsed[i]:
                self.moveForward()
                break
        
        if len(bits) == 1:
            self.positions[self.currentPosition][bits[0]] = gate
            self.currentBitsUsed[bits[0]] = True
            self.size += 1
        # currently only works for two bit gates
        else:
            firstBit = bits[0]
            secondBit = bits[1]
            swaplist = []
            # add swap gates in until the original gate is acting on two sequential bits
            while secondBit-firstBit != 1:
                if firstBit < secondBit:
                    self.add_gate(g.Swap(), firstBit, firstBit+1, advance=True)
                    swaplist.append((firstBit, firstBit+1))
                    firstBit += 1 
                elif secondBit < firstBit:
                    self.add_gate(g.Swap(), secondBit, secondBit+1, advance=True)
                    swaplist.append((secondBit, secondBit+1))
                    secondBit += 1
                # for an 'upside down' gate (e.g. cnot(4, 0)) once we have added all the swaps firstBit == secondBit
                else:
                    firstBit -= 1

            bits = [firstBit, secondBit]
            self.positions[self.currentPosition][bits[0]] = gate
            for bit in bits:
                self.currentBitsUsed[bit] = True
            self.positions[self.currentPosition][bits[1]] = g.Blank() #to make sure we don't over tensor product
            self.size += 1
            if advance:
                self.moveForward()
            if swaplist != []: # swap all bits back to how they were originally if we swapped them
                swaplist.reverse()
                for swapbits in swaplist:
                    self.add_gate(g.Swap(), swapbits[0], swapbits[1])


    def add_gates(self, gates, bits):
        '''
        gates: the gates to add
        bits: the qubits each gate applies to
        Note gates must be added in order from left to right but not necessarily in vertical order
        For example consider circuit with the 2 qubit CNOT gate and some other arbitrary 1 qubit gates A, B, ...
        ---A---.---B---
               |
        ---C---+---D---

        -------E---F---
        This could be implented by:

        add_gates(gates=[A, C], 0,1)
        add_gates(gates=[CNOT, E], 0,1,2)
        add_gates(gates=[B, D, F], 0,1,2)

        But could equally correctly be implemented by:

        add_gates(gates=[A, C], 0,1)
        add_gates(gates=[E, CNOT], 2,0,1)
        add_gates(gates=[B, F, D], 0,2,1)
        '''
        for i in range(len(gates)):
            gate = gates[i]
            gate_size = int(np.log2(gate.matrix.rows))  # number of qubits gate acts on
            gate_bits = []
            for j in range(gate_size):
                gate_bits.append(bits[j])

            for j in range(gate_size):
                self.add_gate(gate, gate_bits)

    def build_circuit(self):
        '''
        Calculates the matrix model by finding the tensor product of each column of gates and then performing the matrix multiplication across the circuit
        '''
        # Identity matrix on top of which the whole matrix implementation is done
        # Note we don't need to do the kronecker product as the kronecker product of two I matrices is just a bigger I
        # so the identity matrix is of size 2^n where n is the number of qubits
        self.matrix = sm.sparseIdentity(2**self.nqubits) #could replace this with the first column tensor product and loop from the second column

        for column in self.positions:
            tpColumn = g.TensorProducts(column)
            self.matrix = sm.multiplySparse(tpColumn, self.matrix)

    def vector_tensor_prod(self, v, w):
        result = []
        for vi in v:
            for wi in w:
                result.append(vi*wi)
        return np.array(result)

    def kronecker_prod(self, V, W):
        nV = len(V)  # V matrix width
        mV = len(V[0])  # V matrix height
        nW = len(W)  # W matrix width
        mW = len(W[0])  # W matrix height
        result = np.zeros((nV * nW, mV * mW))
        counter1 = 0
        for i in range(nV):
            counter2 = 0
            for j in range(mV):
                for k in range(nW):
                    for l in range(mW):
                        result[k + counter1, l + counter2] = V[i, j] * W[k, l]
                counter2 = (j + 1) * mW
            counter1 = (i + 1) * nV
        return result

    def apply(self, qubits:sm.SparseMatrix):
        '''
        qubits: [|psi_1>, |psi_2>, ...] where |psi_i> = a|0> + b|1> s.t. a^2 + b^2 = 1 
        apply the circuit to a certain set of qubits
        '''
        # convert qubit list to sparse matrices 

        initial_state = qubits[0]
        for i in range(1, len(qubits)):
            initial_state = g.TensorProduct(initial_state, qubits[i])  # implement self.vector_tensor_prod(a, b)
        return sm.multiplySparse(self.matrix, initial_state)


if __name__ == '__main__':
    H = g.Hadamard()
    S = g.SGate()
    CNOT = g.Cnot()
    R = g.Rot(np.pi / 4)
    Z = g.ZGate()

    nqubits = 2
    qc = QuantumCircuit(nqubits)

    # The bits should be now lists: [0, 1], [0, 1, 2], ...
    qc.add_gates([H, S], [0, 1])
    qc.add_gate(CNOT, [0, 1])
    qc.add_gates([H, S], [0, 1])

    qc.build_circuit()

    res = qc.matrix.toNPArray()

