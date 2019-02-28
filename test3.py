#良恶性肿瘤
#699条样本，1列用于检索id，9列与肿瘤相关的医学特征（数值介于1-10），肿瘤类型（2：良性，4：恶性），数据包含16个缺失值

import  numpy as np
import  pandas as pd

#创建特征列表
column_names=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

#从网络读取数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)#names参数是用于结果的列名列表

#把？替换为标准缺失值
data = data.replace(to_replace='?',value=np.nan) #NAN

#只要有任何一个维度有缺失就丢弃这个数据
data = data.dropna(how='any')

# print(data.shape) #输出(683, 11)

#分隔数据，75%是训练集，25%是测试集
from sklearn.cross_validation import train_test_split
#参数依次：1-9是样本特征，10是标签，样本占比，随机数
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

# print(y_train.value_counts())
# 2    344
# 4    168
# print(y_test.value_counts())
# 2    100
# 4     71
#得到训练样本512条数据（344良，168恶），测试样本171条数据（100良，71恶）


#使用逻辑回归和随机梯度下降
from sklearn.preprocessing import StandardScaler  #StandardScaler数据标准化预处理
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#标准化数据，保证每个维度特征数据方差为1，均值为0，使预测结果不会被某些维度过大的特征值干扰
ss = StandardScaler()
X_train = ss.fit_transform((X_train))
X_test = ss.transform(X_test)

from sklearn.metrics import classification_report  #分类预估评价函数 精确率、召回率、F1等

lr = LogisticRegression()
lr.fit(X_train,y_train)  #训练模型
lr_y_predit = lr.predict(X_test)  #利用模型预测
# print(lr_y_predit)
print('lr accuracy:',lr.score(X_test,y_test))  #得到精确率：0.9883040935672515
print(classification_report(y_test,lr_y_predit,target_names=['Benign', 'Malignant']))

sgdc = SGDClassifier()
sgdc.fit(X_test,y_test)
sgdc_y_predit = sgdc.predict(X_test)
# print(sgdc_y_predit)
print('sgdc accuracy:',sgdc.score(X_test,y_test))  #得到精确率：0.9824561403508771
print(classification_report(y_test,sgdc_y_predit,target_names=['Benign', 'Malignant']))





