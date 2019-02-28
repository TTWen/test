#良恶性肿瘤预测 示例

import pandas as pd
df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')#传入训练文件数据存入df_train
df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')#传入测试文件数据存入df_test

df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness', 'Cell Size']]#选取肿瘤厚度和大小作为特征，构建测试集中的正负分类样本
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')#绘制良性肿瘤样本点，标记为红色的o
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')#绘制恶性肿瘤样本点，标记为黑色的x

plt.xlabel('Clump Thickness')#绘制x,y轴的说明
plt.ylabel('Cell Size')

plt.show() #p1,训练样本

import numpy as np
intercept = np.random.random([1])#使用random函数随机采样直线的截距和系数
coef = np.random.random([2])
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c="green")#绘制随机直线

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c="red")
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel('Clump Thickness')#绘制x,y轴的说明
plt.ylabel('Cell Size')

plt.show() #p2，训练样本和随机直线

from sklearn.linear_model import LogisticRegression#导入逻辑回归分类器
lr = LogisticRegression()

lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])#使用前10条训练样本训练直线的系数和截距
print('10 samples:%f' %lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))  #输出：0.8685714285714285

intercept = lr.intercept_
coef = lr.coef_[0, :]

ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()  #p3，使用前十个训练样本训练直线的截距和系数

lr = LogisticRegression()

lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print('all samples:%f'%lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])) #模型准确率

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()  #p4，使用所有训练样本训练直线的截距和系数

