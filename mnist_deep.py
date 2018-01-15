import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

return base.Datasets(train=train, validation=validation, test=test)

Datasets is a collection.namedtuple type
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])



"""

x = tf.placeholder(tf.float32, shape =[None, 784]) #wer create this to feed in the training images
y_ = tf.placeholder(tf.float32, shape = [None, 10])

sess = tf.InteractiveSession()

"""my exercise now is to use the validation set as training and use the test set as test
1. Use a bunch of different types of optimizers and test the difference .
2. Use a bunch of different types of learners and see the difference
3. Test how big the none datatype is when assigned for different batch sizes maybe by print statements
4. test this without the shape argument"""


"""The above placeholders shouldn't change no matter what data I use to train
The none is a placeholder

Now we are usign the tensorflow variable for W and b
"""

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

#loss = tf.reduce_sum(tf.square(y - y_))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

"""Here is where to change the optimizer """

train_step = tf.train.GradientDescentOptimizer(.01).minimize(loss)


for _ in range(1000):
  batch = mnist.train.next_batch(100)
  #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

