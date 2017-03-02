# -*- coding: utf-8 -*-
# @Time    : 2017/2/18 16:51
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

'''
Handwirting    加载数据并用神经网络预测结果
'''

import pandas as pd
from sklearn import preprocessing
import time

#加载训练数据和训练标签
print '加载训练数据文件...\n'
pre_train_data = pd.read_csv('train.csv')
train_label = pre_train_data['label'];del pre_train_data['label']
train_data = pre_train_data

#加载测试数据
print '加载测数据文件...\n'
test_data = pd.read_csv('test.csv')

#均值归一化 训练数据和测试数据
print '分别对训练数据和测试数据进行均值归一化...\n'
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
training_data = pd.DataFrame(min_max_scaler.fit_transform(train_data)) #numpy的ndarray类型
testing_data = pd.DataFrame(min_max_scaler.fit_transform(test_data))

#神经网络
def neural_network_classifier(train_x, train_y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(150,50,20), alpha=0.1)
    model.fit(train_x, train_y)
    return model

print '训练开始计时...\n'
start_time= time.time()
neural_network_model = neural_network_classifier(training_data, train_label)
test_Pre = pd.DataFrame(neural_network_model.predict(testing_data), columns=['Label'])

end_time = time.time()
print '训练结束...花费 %d 秒\n' % (end_time - start_time)


index = pd.DataFrame( range(1,test_data.shape[0]+1), columns=['ImageId'])
final_data = pd.merge(index, test_Pre, left_index=True, right_index=True, how='outer')
final_data.to_csv('result3.csv', index=False)
print '预测结果保存完毕.'

