#001-Tensorflow中的数据节点(node)
tensorflow中有三种形式的数据node，可以是任意形式的tensor(即scalar, list, matrix等)。

* constant: 常量
* placeholder：占位，可用于inputs
* variable：一般通过get_variable(name, shape)的形式声明，可用于模型参数
 
##1 constant
```
import tensorflow as tf

const1 = tf.constant([[1,2,3], [1,2,3]]);
const2 = tf.constant([[3,4,5], [3,4,5]]);

result = tf.add(const1, const2);

with tf.Session() as sess:
  output = sess.run(result)
  print(output)

```

##2 placeholder 

placeholder最后通过feed_dict给予其值。

```
parameter = tf.placeholder(tf.int32)
sum = two + parameter
sess = tf.Session()
sess.run(sum, feed_dict={parameter:0.0001})
```

##3 Variable
###3.1 Variable

```
k = tf.Variable(tf.zeros([1]), name="k")
k = tf.Variable(tf.add(a, b), trainable=False)
```

```
import tensorflow as tf

var1 = tf.Variable([[1, 2], [1, 2]], name="variable1")
var2 = tf.Variable([[3, 4], [3, 4]], name="variable2")

result = tf.matmul(var1, var2)

with tf.Session() as sess:
  output = sess.run(result)
  print(output)
```

```
import tensorflow as tf

x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="a", dtype=tf.float32)
b = tf.constant(13.0, name="b", dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print session.run(y)
```

###3.2 `get_variable` with a `initializer`

在这种情况下，需要`tf.global_variables_initializer()`

```
import tensorflow as tf
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count", [], initializer=const_init_node)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print sess.run(count_variable)
```

###3.3 `get_variable` then `assign`

```
import tensorflow as tf
count_variable = tf.get_variable("count", [])
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess = tf.Session()
sess.run(assign_node)
print sess.run(count_variable)
```




