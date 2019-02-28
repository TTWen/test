#决策树：对泰坦尼克号乘客是否生还的预测可能性

#数据集包含1313条乘客信息，存在部分特征缺失

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# titanic.head()
# titanic.info()

#选择特征
X = titanic[['pclass','age','sex']]
y = titanic['survived']
#对当前选择的特征进行侦查
X.info()#pclass和sex都是1313条，且都是类别型的，需要转化为数值型用0/1代替，age数据缺少，只有633条，需要补充

#补充age里的数据，使用平均数或中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(),inplace = True)
#补充完重新侦查数据完整性
# X.info()

#数据分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

#使用特征转换器进行特征抽取
#转换特征后，类别型的特征都单独剥离出来，独成一系列值，数值型的则保持不变
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_) ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']

#对测试数据也进行转换
X_test =  vec.transform(X_test.to_dict(orient='record'))

#导入决策树分类器
from sklearn.tree import DecisionTreeClassifier#导入
dtc = DecisionTreeClassifier() #默认初始化
dtc.fit(X_train,y_train)  #学习
y_predict = dtc.predict(X_test)  #预测
print("accuracy:",dtc.score(X_test,y_test))  #0.7811550151975684

#报告
from sklearn.metrics import classification_report
print(classification_report(y_predict,y_test,target_names=['died','survived']))










