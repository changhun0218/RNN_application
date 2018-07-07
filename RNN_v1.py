import tensorflow as tf
import numpy as np
import os

def make_batch(batch_size, n_time_steps, data):
    input_batch = []
    output_batch = []
    for _ in range(batch_size):
        rnum = np.random.randint(5000)
        input_batch.append(data[rnum:rnum + n_time_steps, 2:6])
        output_batch.append(data[rnum + n_time_steps: rnum + 2 * n_time_steps, 2:6])
    return input_batch, output_batch
    
    
cwd = os.getcwd()
inputs = np.load(cwd + "/RNN_data/input.npy")
outputs = np.load(cwd + "/RNN_data/output.npy")

n_inputs = 4
n_outputs = 4
n_neurons = 100
n_layers = 2
n_time_steps = 10

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_time_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_time_steps, n_outputs])

layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

basic_cell = tf.contrib.rnn.OutputProjectionWrapper(multi_layer_cell, output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

learning_rate = 0.01
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(10000):
        input_batch, output_batch = make_batch(batch_size, n_time_steps, inputs)
        sess.run(train_op, feed_dict={X: input_batch, y: output_batch})
        if iteration % 100 == 0:
            print(iteration, loss.eval(feed_dict={X: input_batch, y: output_batch}))     

    y_pred = sess.run(outputs, feed_dict={X: inputs[300:310,2:6].reshape(-1,10,4)})
    print(loss.eval(feed_dict={X: inputs[300:310,2:6].reshape(-1,10,4), y:inputs[310:320,2:6].reshape(-1, 10, 4)}))

print(inputs[300:310, 2:6])
print(inputs[310:320, 2:6])
print("prediction :", y_pred)
