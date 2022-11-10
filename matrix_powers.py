import numpy as np 
import matplotlib.pyplot as plt

def helper(eig, t, k):
    matrix = np.zeros((t, t))
    matrix[0][0] = 1
    matrix[0][-1] = eig
    for i in range(1, t):
        matrix[i][i-1] = 1
    return np.linalg.matrix_power(matrix, k)[0][0]

def polynomial(eigs, t, k):
    # diagonal with helper values 
    diagonal = [helper(eig, t, k) for eig in eigs]
    return np.diag(diagonal)
    
if __name__ == '__main__':
    eigs = [-0.13]
    vals = []
    size = 500
    for i in range(size):
        vals.append(polynomial(eigs, 16, i)[0][0])
    plt.plot(range(size), vals)
    plt.title('tau = 16, x = 0.13')
    plt.savefig('oscillations.png')