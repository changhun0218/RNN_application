import tensorflow as tf
import numpy as np

inputs = np.load("../RNN_data/input.npy")
outputs = np.load("../RNN_data/output.npy")

#n1553=inputs[np.where(inputs[:,0]==1553)[0]]
#n1553.shape

final_input = []
y_ = []

for i in range(10):
    rnum = i*20
    temp = inputs[rnum:rnum+10,2:6]
    final_input.append(temp)
    y_.append(inputs[rnum+10:rnum+20,2:6])
final_input = np.array(final_input)
y_ = np.array(y_)

"""
print(final_input.shape)
print(y_.shape)
print(final_input)
print(y_)

print(inputs[300:310,2:6])
"""

del outputs, states

n_inputs = 4
n_outputs = 4
n_neurons = 200
n_time_steps = 10

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_time_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_time_steps, n_outputs])
basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)


# In[52]:


learning_rate = 0.01
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for _ in range(100):
        sess.run(train_op, feed_dict={X:final_input, y:y_})
        
    print(loss.eval(feed_dict={X: final_input, y:y_}))
    y_pred = sess.run(outputs, feed_dict={X: inputs[300:310,2:6].reshape(-1,10,4)})


print(inputs[300:310,2:6])
print(y_pred)

