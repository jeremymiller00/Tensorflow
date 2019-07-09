import tensorflow as tf 

# bias weights
b= tf.Variable(tf.zeros((100,)))

# weight matrix
W = tf.Variable(tf.random_uniform((784, 100), -1, 1))


# placeholder for data
x = tf.placeholder(tf.float32, (100, 784))

# graph
h = tf.nn.relu(tf.matmul(x, W) + b)