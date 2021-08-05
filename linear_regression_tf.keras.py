import tensorflow as tf
from publicMethod import *

test_flag = False
file_name = "data/GSOD_2021/583620-99999-2021.op"
train_set = 175
tem_avg, tem_min, tem_max = open_file(file_name)
(train_data, train_label), (test_data, test_label) = generate(train_set, tem_avg)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(1, activation="relu")
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse')
model.summary()
model.fit(train_data, train_label, batch_size=10, epochs=100)

model.evaluate(test_data, test_label)
print(model.predict([tem_avg[-7:]]))

tem_predict = tem_avg[:7]
for i in range(len(tem_avg) - 7):
    tem_predict.append(model.predict([tem_avg[-7:]])[0][0])


if test_flag:
    import matplotlib.pyplot as plt
    print(tem_avg, "\n", tem_max, "\n", tem_min)
    plt.plot(tem_avg)
    plt.plot(tem_min)
    plt.plot(tem_max)
    plt.plot(tem_predict)
    plt.show()
