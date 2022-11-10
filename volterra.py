import tensorflow as tf 
import random 
import matplotlib.pyplot as plt
from tqdm import tqdm

with tf.device('/device:GPU:0'):
    gamma = 1.2 
    delta = 0.1
    n = 5000
    d = int(n / 0.9)
    A = tf.random.normal([n, d])
    x_tilde = tf.random.normal([d, 1])
    eta = tf.random.normal([n, 1])
    b = A @ x_tilde + eta * tf.sqrt(float(n))  

    def update(x):
        global grads
        coord = random.randint(0, n-1)
        a_i = A[coord]
        b_i = b[coord][0]
        inner = tf.tensordot(a_i, x, 1) - b_i
        scaled = gamma/float(n) * tf.expand_dims(a_i, 1) * inner
        grads.append(scaled)
        # random.shuffle(grads)
        scaled = grads[0]
        grads = grads[1:]
        return x - scaled

    init = tf.Variable(tf.random.normal([d, 1]))

    for delay in [0, 8, 16, 32, 64, 128, 256, 512, 1024]:
        grads = [x_tilde*0]*delay
        losses = []
        epochs = 5
        x = tf.Variable(init)
        for i in tqdm(range(n*epochs)):#*(delay+1))):
            x = update(x)
            # if i % (delay+1) == 0:
            loss = 1/float(n*2) * tf.linalg.norm(A @ x - b)**2 
            losses.append(loss)
        plt.plot(losses, label=f"delay={delay}")
        plt.legend()
        plt.savefig('volterra.png')