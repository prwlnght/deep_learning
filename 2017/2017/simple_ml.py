import tensorflow as tf

"""The goal of this program is to wirte a simple gradient desecent based learned in tensorflow"""

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model placeholders
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
linear_model = W * x + b


#define loss
loss = tf.reduce_sum(tf.square(linear_model - y))
#define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


#training data
x_train = [1, 2, 3, 4]
y_train = [0, 1, 2, 3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)#init with random vales given

for i in range (1000):
    sess.run(train, {x:x_train, y:y_train})

#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print('curr_loss %s' %curr_loss)

