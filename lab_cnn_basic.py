import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()

image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

print("image.shape : ", image.shape)
# (1, 3, 3, 1) -> (a, b, c, d)
#  a: 몇개의 image를 이용할 것인지?
#  b, c: b x c 의 이미지를 이용한다.
#  d : filter를 몇개 사용할것인지?
plt.imshow(image.reshape(3, 3), cmap='Greys')

# Filter : 2, 2, 1, 1 -> 2 x 2의 image, color는 1가지, 1개의 filter 왼쪽부터 차례대로
weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]])

print("weiht.shape: ", weight.shape)
# padding = 'SAME' 입력과 출력의 크기를 같게 한다!
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d_img.shape: ", conv2d_img.shape)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 2, i+1), plt.imshow(one_img.reshape(3, 3), cmap='gray')