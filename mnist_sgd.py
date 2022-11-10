import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import random
from matrix_powers import polynomial
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from mnist import x_train, y_train

with tf.device('/gpu:0'):
    n = 6000
    d = 784
    batch_size = 100
    tau = 6
    iterations = 20
    gamma = 0.01

    A = tf.constant(x_train[:n])
    b = tf.constant(y_train[:n])
    b = b / tf.norm(b)
    b = tf.reshape(b, (n, 1))

    zeta = batch_size/n

    def loss(x):
        return tf.norm(A @ x - b)**2

    def project(batch_size):
        indices = list(range(n))
        random.shuffle(indices)
        diagonal = [1 if indices.index(i) < batch_size else 0 for i in range(n)]
        return tf.linalg.diag(diagonal)

    for j in range(3):
        x_es = []
        for i in range(tau):
            x_es.append(tf.zeros((d, 1)))
        losses = []
        for i in range(iterations):
            P = project(batch_size)
            gradient = A.T @ P @ (A @ x_es[0] - b)
            x_es = x_es[1:] + [x_es[-1] - gamma * gradient]
            print(loss(x_es[-1]))
            losses.append(loss(x_es[-1]))

        plt.plot(list(range(iterations)), losses, label=f"stochastic {j}")
        print()
    # ====================================================
    # Utils 
    eigh_cache = {}
    def eigh(H):
        # hash H 
        key = hash(str(H.numpy()))
        if key in eigh_cache:
            return eigh_cache[key] 
        eigs, U = tf.linalg.eigh(H)
        eigh_cache[key] = eigs, U
        return eigs, U
    # ====================================================
    # block matrix [[A, 1], [1, 0]]

    Hess = A.T @ A 
    pow_cache = {}
    def poly(H, k, tau):
        H = -1 * gamma * zeta * H
        if (k, tau) in pow_cache:
            return pow_cache[(k, tau)]

        eigs, U = eigh(-1 * H)
        eigs *= -1 
        """
        def power(a, k):
            d = np.sqrt(4 * a + 1)
            l_1 = (1-d)/2
            l_2 = (1+d)/2
            return 1/d * np.array([
                [l_2**(k+1) - l_1**(k+1), a*(l_2**k - l_1**k)],
                [l_2**k - l_1**k, a*(l_2**(k-1) - l_1**(k-1))]
            ])

        one = []
        two = []
        three = []
        four = []
        for i in eigs:
            p = power(i, k)
            one.append(p[0][0])
            two.append(p[0][1])
            three.append(p[1][0])
            four.append(p[1][1])
        
        D = np.zeros((d, 2))

        D[:d, :d] = np.diag(one)
        D[:d, d:] = np.diag(two)
        D[d:, :d] = np.diag(three)
        D[d:, d:] = np.diag(four)"""

        D = polynomial(eigs, tau, k)

        pow_cache[(k, tau)] = U, D
        return U, D


    def get_x(k, tau):
        k = k-1
        acc = tf.zeros((d, 1))
        for t in range(1, k+1):
            U, D = poly(Hess, k-t, tau)
            acc += gamma * zeta * U @ (D @ (U.T @ (A.T @ b)))
        return acc 

    losses = []
    for k in range(2, iterations+2):
        x = loss(get_x(k, tau))
        losses.append(x)
        print(x)

    plt.plot(list(range(iterations)), losses, label="deterministic (full batch)")

    # ====================================================
    psi_cache = {}
    def psi(k, tau):
        if (k, tau) in psi_cache:
            return psi_cache[(k, tau)]
        k = k-1
        acc = 0 

        eigs, U_0 = eigh(Hess)
        for t in range(1, k+1):
            U, D = poly(Hess, k-t, tau)
            G = tf.linalg.diag_part(D)**2 * eigs**2
            trace = tf.math.reduce_sum(G)
            acc += 1/n * gamma**2 * (zeta - zeta**2) * trace * psi(t, tau)
        psi_cache[(k+1, tau)] = acc + loss(get_x(k, tau))
        return acc + loss(get_x(k, tau))

    print()
    losses = []
    for k in range(3, iterations+3):
        x = psi(k, tau)
        losses.append(x)
        print(x)

    plt.plot(list(range(iterations)), [i for i in losses], label="prediction for stochastic")
    plt.title(f"Delay = {tau-1}, Batch Size = {batch_size}")
    plt.legend()
    plt.savefig(f"n={n},d={d},batch_size={batch_size},gamma={gamma}.png")