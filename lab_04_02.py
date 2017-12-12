import numpy as np
import tensorflow as tf
tf.set_random_seed(777)

## csv 파일을 이용해서 data를 받아온다.
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
## 읽어올 파일이 너무 많아 메모리에 부담이 될 때, Queue Runners
## 1. filename_queue = tf.train.string_input_producer([filename, , ...], shuffle = False, name='filename_queue')
## 2. reader = tf.TextLineReader()
##     key, value = reader.read(filename_queue)
## 3. record_defaults =[[0.], [0.]]
##    xy = tf.decode_csv(value, record_defaults = record_defaults)

## numpy 쓰는 방법을 더 공부해둘 필요.
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

## 행렬곱 matmul
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

## 이렇게 학습 시킨 뒤에 다른 X data를 주면 hypothesis가 예측을 해준다.