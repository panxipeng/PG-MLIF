# import tensorflow as tf
#
# print(tf.__version__) #2.4.1
#
# # 生成训练数据( 随便构建了一个线性函数：y = 2x + 5 )
# x_data = tf.reshape(tf.range(0, 100, dtype=tf.float32), (100, 1))
# y_data = x_data * 2 + 5
#
# # 构建keras模型，并训练
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
# model.compile(optimizer="adam", loss="mse")
# model.fit(x_data, y_data, epochs=5000, verbose=1)
#
# # 保存模型
# tf.saved_model.save(model, "tf241_model")
#
# # 测试模型效果
# y_pre = model.predict([10, 20])
# print(y_pre)
