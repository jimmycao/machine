import tensorflow as tf
import random

### build the graph
## first set up the parameters
w = tf.get_variable("m", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))

## then set up the computations
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_guess = w * x + b
loss = tf.square(y - y_guess)

## finally, set up the optimizer and minimization node
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

### start the session
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)


## set up problem
true_w = random.random()
true_b = random.random()

for update_i in range(10):
  ## (1) get the input and output
  input_data = random.random()
  output_data = true_w * input_data + true_b

  ## (2), (3), and (4) all take place within a single call to sess.run()!
  _loss, _ = sess.run([loss, train_op], feed_dict={x: input_data, y: output_data})
  print update_i, _loss

### finally, print out the values we learned for our two variables
print "True parameters:     w=%.4f, b=%.4f" % (true_w, true_b)
print "Learned parameters:  w=%.4f, b=%.4f" % tuple(sess.run([w, b]))
