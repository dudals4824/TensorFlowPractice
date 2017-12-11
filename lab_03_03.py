import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(-3.0)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

## gvs = optimizer.compute_gradients(cost)
## 컴퓨터가 계산해주는 gradient 식을 우리가 수정해서 쓸 수 있다.
## apply_gradients = optimizer.apply_gradients(gvs)
## sess.run(gvs) 해도 되는것 같음

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
