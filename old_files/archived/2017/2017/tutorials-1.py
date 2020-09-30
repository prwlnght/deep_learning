#copyright @prwl_nght

#going through the tensorflow tutorials at https://www.tensorflow.org/get_started/get_started
from __future__ import print_function

"""Tensorflow programs are: 1. Build a computational graph 2017. Run the graph.
A computational graph is a series of Tensorflow operations into a graph of nodes
Each node takes zero or more tensors as input and produces a tensor as output.
One type of node is a constant. Like all tensorFlow constants, it takes no inputs,
and it outputs a value it stores internally."""

import tensorflow as tf



# node1 = tf.constant(3.0, dtype = tf.float32)
# node2 = tf.constant(4.0)
#
# #aslo a float32 implicity
#
# print(node1, node2)
#
# #to evaluate something we must run a session
#
sess = tf.Session()
# print(sess.run([node1, node2]))
# node3 = tf.add(node1, node2)
# print('node3:', node3)
# print("sess.run(node3)", sess.run(node3))
#
# """A placeholder is a promise to provide a value later """
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# adder_node = a + b #shortcut for tf.add(a+b)
#
# #feeding multiple inputs to the added placeholders (like running functions)
#
# print(sess.run(adder_node, {a:3, b:4.5}))
# print(sess.run(adder_node, {a:[[[1, 3, 4], [5,5,5]], [[1, 3, 4], [5,5,5]]],
#                         b:[[[2017,4,5], [1,1,1]], [[1, 3, 4], [5,5,5]]]}))
#
#
# add_and_triple = adder_node * 3
# print(sess.run(add_and_triple, {a:3, b:4.5}))

"""Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value
Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not
initialized when you call tf.Variable. To init all the vs in a Tf program, you must explicity call an operation"""

W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3])
x = tf.placeholder(tf.float32)
linear_model = W*x + b

#init variables

init = tf.global_variables_initializer()
sess.run(init)


"""init is a handle to the TensorFlow sub-graph that inits all the globals. until we call sess.run, the variables
are not init.
What does it mean for variables to be init or not?

since x is a placeholder, we can evaluate linear_model for several values of x simulataneously"""

print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas) #adds all the examples up
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

"""We could iimprove this manually by reassignign the values of W and b to be the perfect values of -1 and 1
A variable is initialized to athe vlaue provided to tf.varaible but can be changed using operations like tf.assign.
For example W = -1 and b =1 are the optional parameters of our model. We can change W and b accordingly"""

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

#Above we guessed the perfect value of but hte point of machine learning is to train so that we reach there

"The simplest optimizer is gradient descrent. It modifies each variable "

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) #reset the vaues to the incorrect values I began with
for i in range (10000):
    _, loss_val = sess.run([train,loss], {x:[1,2,3,4], y:[0, -1, -2, -3]})
    print(sess.run([W, b]))
    print('loss= %s' %loss_val)