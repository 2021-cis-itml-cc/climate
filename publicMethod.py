import math
import numpy as np
import time
import tensorflow as tf

print("Import publicMethod")

class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


def generate(train_set, data_set):
    """train_set: training set 的大小，
    data_set: 传入list, 数据本身"""
    test_set = len(data_set) - 8

    def get_data(set1, set2, data_set):
        train_data = []
        train_label = []
        for i in range(set1, set2):
            train_data.append(data_set[i:i + 7])
            train_label.append(data_set[i + 8])
        train_data = np.array(train_data)
        train_label = np.array(train_label).T
        return train_data, train_label

    train_data, train_label = get_data(0, train_set, data_set)
    test_data, test_label = get_data(train_set, test_set, data_set)

    return (train_data, train_label), (test_data, test_label)


def open_file(file_name, show_index = False):
    """打开op文件并将温度信息抽取"""
    with open(file_name) as f:
        data_read = f.read().split("\n")
    tem_avg, tem_max, tem_min, index = [], [], [], []
    for i in data_read[1:-1]:
        tem_avg.append(float(i[25:30]))
        tem_max.append(float(i[103:108]))
        tem_min.append(float(i[111:116]))
        index.append([i[14:22], file_name])

    return (tem_avg, tem_min, tem_max, index) if show_index else (tem_avg, tem_min, tem_max)


def normal(x, mu, sigma):
    """正态分布定义器"""
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

def linear_regression(x, params):
    return tf.matmul(params,(x ** np.arange(len(params)))) # which is obviously y

if __name__ == '__main__':
    tem_avg, tem_min, tem_max = open_file("data_op/722860/722860-23119-1933.op")
    print(max(tem_max))