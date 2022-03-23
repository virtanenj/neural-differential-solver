'''
Copied from sparse_matrix.py and modifies for the dictionary key represenation:
    {(i,j): value}
where (i,j) is the location of value in the matrix. 
'''


import numpy as np

ROW = 0
COLUMN = 1
VALUE = 2

class SparseMatrix:
    def __init__(self, matrix = 0, rows = 0, columns = 0, empty = False, sparse = False):
        '''
        See https://en.wikipedia.org/wiki/Sparse_matrix#Dictionary_of_keys_(DOK)
        '''
        if empty:
            self.rows = rows
            self.columns = columns
            self.matrix = {}
        elif sparse:
            self.rows = rows
            self.columns = columns
            self.matrix = matrix
        else:
            self.rows, self.columns = matrix.shape
            self.matrix = {}
            for i in range(self.rows):
                for j in range(self.columns):
                    if matrix[i][j] != 0:
                        self.matrix[(i,j)] = matrix[i][j]

    def __eq__(self, other: object) -> bool:
        # Might be faster to not convert into Numpy array and just compare each (i, j, non-zero value)
        if isinstance(other, self.__class__):
            return self.toNPArray().all() == other.toNPArray().all()
        else:
            return False
    
    def getShape(self):
        return (self.rows, self.columns)

    def toNPArray(self):
        NPMatrix = np.zeros((self.rows, self.columns), dtype=complex)
        for key in self.matrix.keys():
            NPMatrix[key[ROW]][key[COLUMN]] = self.matrix[key]
        return NPMatrix

    def clear(self):
        self.matrix = {}

    def getLocation(self, location:int):
        '''
        location: the item in the list to get the position of
        helper method that gets the position in the full size matrix corresponding to the position in the sparse matrix
        '''
        # return (self.matrix[location][ROW], self.matrix[location][COLUMN])
        return None

    def transpose(self, inplace=True):
        '''
        inplace: whether to do the transpose and overwrite the current matrix, or return a new one
        find the transpose of the matrix
        '''
        if inplace:
            temp = self.rows
            self.rows = self.columns
            self.columns = temp
            newMatrix = {}
            for key in self.matrix.keys():
                newMatrix[(key[1], key[0])] = self.matrix[key]
            self.matrix = newMatrix
        else:
            newMatrix = {}
            for key in self.matrix.keys():
                newMatrix[(key[1], key[0])] = self.matrix[key]
            return SparseMatrix(matrix=newMatrix, rows=self.columns, columns=self.rows, sparse=True)
        return


def addSparse(firstMatrix: SparseMatrix, otherMatrix: SparseMatrix):
    pass

def tensorSparse(firstMatrix: SparseMatrix, otherMatrix: SparseMatrix):
    newMatrix = SparseMatrix(empty=True, rows=firstMatrix.rows*otherMatrix.rows, columns=firstMatrix.columns*otherMatrix.columns)
    newMatrixDict = {}
    s = firstMatrix.rows
    m = firstMatrix.columns
    for blj in otherMatrix.matrix:
        j = blj[ROW] + 1
        l = blj[COLUMN] + 1
        b = otherMatrix.matrix[(j - 1, l - 1)]
        # b = blj[VALUE]
        for aki in firstMatrix.matrix:
            i = aki[ROW] + 1
            k = aki[COLUMN] + 1
            a = firstMatrix.matrix[(i - 1, k - 1)]
            # a = aki[VALUE]

            lk = (l - 1) * s + k - 1
            ji = (j - 1) * m + i - 1
            
            newMatrixDict[(lk, ji)] = b * a
    newMatrix = SparseMatrix(matrix=newMatrixDict, rows=firstMatrix.rows*otherMatrix.rows, columns=firstMatrix.columns*otherMatrix.columns, sparse=True)
    return newMatrix

def sparseDotProduct(firstMatrix:SparseMatrix, otherMatrix:SparseMatrix, offset = 0):
    pass

def multiplySparse(firstMatrix: SparseMatrix, otherMatrix: SparseMatrix):
    '''
    c_ij = sum_k^n a_ik b_kj
    '''
    if firstMatrix.columns != otherMatrix.rows:
        raise ValueError('matrices must be of form (n,k) (k,m) to multiply. Cannot multiply matrices with shapes: ', firstMatrix.getShape(), otherMatrix.getShape())
    A = firstMatrix
    B = otherMatrix
    newMatrixDict = {}
    for keyA in A.matrix.keys():
        i, k = keyA
        for j in range(B.columns):
            if (k, j) in B.matrix.keys():
                keyB = (k, j)
                if (i, j) not in newMatrixDict.keys():
                    newMatrixDict[(i, j)] = A.matrix[keyA] * B.matrix[keyB]
                else:
                    newMatrixDict[(i, j)] += A.matrix[keyA] * B.matrix[keyB]
    return SparseMatrix(matrix=newMatrixDict, rows=firstMatrix.rows, columns=otherMatrix.columns, sparse=True)

def multiplySparseVec(matrix: SparseMatrix, vector:np.array):
    '''
    matrix: SparseMatrix
    vector: 1 dimensional numpy array
    '''
    if matrix.columns != len(vector):
        raise Exception('matrix and vector are not correct shape')

    newVector = np.zeros(matrix.rows)
    for i in range(matrix.columns):
        for j in range(matrix.rows):
            if (j,i) in matrix.matrix.keys():
                newVector[j] += matrix.matrix[(j,i)] * vector[i]
    return newVector


def sparseIdentity(n):
    # return SparseMatrix([(i,i,1) for i in range(n)], rows=n, columns=n, sparse=True)
    pass
