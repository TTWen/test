#线性回归器预测美国波士顿房价

import  numpy as np
#导入房价数据
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.DESCR)  #共有506条房价数据，每条数据包括13项对指定房屋的数值型特征描述和目标房价，没有丢失的属性

X = boston.data
y = boston.target
print("boston-shape",np.shape(boston))
print("X-shape:",np.shape(X))  #(506, 13)
print("y-shape:",np.shape(y))  #(506,)

#数据分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
print("X_train-shape:",np.shape(X_train))


#分析回归目标值的差异
# print("the max target is :",np.max(boston.target))  #50.0
# print("the min target is :",np.min(boston.target))  #5.0
# print("the mean target is :",np.mean(boston.target))  #22.532806324110677
#发现预测目标 房价之间差异大，需要对特征及目标值进行标准化处理

#标准化处理
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train.reshape(-1,1))
X_test = ss_X.transform(X_test.reshape(-1,1))
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

#使用线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

#使用SGD回归
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)

#评价方法：使用线性回归自带的评估模块
print("the value of default measurement of LinearRegression is:",lr.score(X_test,y_test))

#平均绝对误差MAE，均方误差MSE，R-squared
#mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("the value of r2 of LinearRegression:",r2_score(y_test,lr_y_predict))
print("the value of mean_aquared_errro of LinearRegression:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print("the value of mean_absolute_error of LinearRegression:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))



















