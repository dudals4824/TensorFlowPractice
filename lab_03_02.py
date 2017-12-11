import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis 모델을 H(x) = W*x 로 잡음
hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

# cost 함수를 미분한 Gradient Descent 함수
# optimizer = tf.triain.GradientDescentOptimizer 이 부분이다.
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))