import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from publicMethod import *
time = Timer()

# get train and test data form op
file_name = "data/GSOD_2021/010010-99999-2021.op"
avg, min, max = open_file(file_name)
(train_data, train_label), (test_data, test_label) = generate(int(0.75*len(avg)), avg)

# build regression model
# linear_regression
params = np.array([random.randint(-1.0, 1.0) for i in range(20)])
x = np.arange(0, 20)
y = np.array(linear_regression(x, params))
plt.plot(x, y)
plt.show()
print(params)

# data iter updater
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i:min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)

# 初始化模型参数
w = tf.Variable(tf.random.normal(shape=(20, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

# 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型。"""
    return linear_regression(X, w) + b
print(linreg(1, w, b))
