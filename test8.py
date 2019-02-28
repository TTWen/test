#使用单一决策树、随机森林分类、梯度上升决策树 对上例泰坦尼克号数据进行处理

#读取数据
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#选取特征
X = titanic[['pclass','age','sex']]
y = titanic['survived']

#填补缺失数据
X['age'].fillna(X['age'].mean(),inplace = True)

#数据分割
from sklearn.cross_validation import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

#特征抽取转换
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

#使用单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)

#使用随机森林分类
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)

#使用梯度上升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

#评估
from sklearn.metrics import classification_report
print("accuracy1:",dtc.score(X_test,y_test)) #0.7811550151975684
print(classification_report(dtc_y_pred,y_test))

print("accuracy2:",rfc.score(X_test,y_test)) #0.7872340425531915
print(classification_report(rfc_y_pred,y_test))

print("accuracy3:",gbc.score(X_test,y_test)) #0.790273556231003
print(classification_report(gbc_y_pred,y_test))











