import tensorflow as tf 
import random
import matplotlib.pyplot as plt

class Objective:
    def __init__(self, n=100, E=100, p=0.07):
        self.components = []    
        self.n = n
        self.E = E
        self.p = p
        self.subfunctions = []
        for i in range(E):
            self.subfunctions.append(self.make_random_subfunction())
    
    def objective(self, x):
        return tf.reduce_sum([f(x) for f in self.subfunctions])

    
    def get_psd_matrix(self, size):
        A = tf.random.normal([size, size])
        return tf.matmul(A, A, transpose_b=True)
    
    def make_random_subfunction(self):
        P = []
        size = 0
        for i in range(self.n):
            if tf.random.uniform([]) < self.p:
                temp = [0]*self.n
                temp[i] = 1 
                P.append(temp)
                size += 1 
        if size == 0:
            return self.make_random_subfunction()
        P = tf.constant(P, dtype=tf.float32)
        A = self.get_psd_matrix(size)
        B = tf.random.normal([size])
        c = tf.random.normal([])
        def subfunction(x):
            e = tf.linalg.matvec(P, x)
            return tf.tensordot(e, tf.linalg.matvec(A, e), 1) + tf.tensordot(B, e, 1) + c
        return subfunction


o = Objective()
x = tf.Variable(tf.random.normal([o.n]))
y = tf.Variable(x)
loss = o.objective
step = tf.constant(0.01) 

# minimize the loss with gradient descent 
hogwild_losses = [] 

grad_array = []
for _ in range(500):
    with tf.GradientTape() as tape:
        tape.watch(x)
        function = random.choice(o.subfunctions)
        loss_value = function(x)
        grads = tape.gradient(loss_value, x)
        grad_array.append(grads)
        if len(grad_array) >= 10:
            print("modifying!")
            random.shuffle(grad_array)
            grad = grad_array.pop()
            x.assign_sub(step * grad) 
        loss_value = loss(x)
        hogwild_losses.append(loss_value)
        print(loss_value)

# get every 10th value in hogwild_losses 
hogwild_losses = hogwild_losses[::10]
print("Sequential SGD")
standard_sgd = []
for _ in range(50):
    with tf.GradientTape() as tape:
        tape.watch(y)
        function = random.choice(o.subfunctions)
        loss_value = function(y)
        grads = tape.gradient(loss_value, y)
        y.assign_sub(step * grads)
        standard_sgd.append(loss(y))

# plot the loss over time
plt.plot(hogwild_losses, label="Hogwild (10 processors)")
plt.plot(standard_sgd, label="Sequential SGD")
plt.legend()
plt.xlabel("Wall clock time")
plt.ylabel("Loss")
plt.savefig("hogwild.png")
# save plot 
