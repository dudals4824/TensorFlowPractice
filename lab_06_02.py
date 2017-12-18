import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
# one_hot 함수는 N차 데이터를 넣으면 N+1차 데이터를 반환해준다.
# 에러없이 이용하려면 reshape 과정이 꼭 필요하다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# sotfmax_cross_entropy_with_logits 를 위한 logits 선언
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
       cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_data, Y: y_data})
       if step % 10 == 0:
           print(step, "Cost: ", cost_val, "\n Prediction: \n", hy_val)


