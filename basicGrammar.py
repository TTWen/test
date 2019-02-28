from math import  sqrt #导入库内指定函数
import numpy as np #导入整个库
print("hello!\n")

# def quicksort(arr): #快排
#     if(len(arr)<=1):
#         return arr
#     pivot = arr[len(arr)/2] #关键值
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quicksort(left) + middle + quicksort(right)
#
# quicksort([3,4,8,1,2,9,6])

# 基本数据类型
# x=3
# print(type(x))  #输出<class 'int'>
# print(x,x+1,x*2,x**2)  #输出3,4,6,9
# x+=1
# print(x)  #输出4，Python里没有x++和x--
# x*=2
# print(x)  #输出8

#布尔型
# t=True
# f=False
# print (type(t))  #输出<class 'bool'>，也支持对bool变量的运算

#字符串
# hello = "hello" #单引号和双引号无所谓
# world = 'world'
# print(hello)
# print(len(hello))  #6
# helloworld = hello +' '+world
# print(helloworld)
# helloworld2 = '%s %s %d'%(hello, world,12)  #按固定格式输出
# print(helloworld2)

#字符串方法
# s="hello"
# print(s.capitalize())  #开头大写
# print(s.upper())  #所有字母大写
# print(s.rjust(7))  #7个字符右对齐
# print(s.center(7))  #7个字符居中对齐
# print(s.replace('l','(ell)'))  #字母替换输出he(ell)(ell)o
# print('   world'.strip())  #去掉空白
# a = 'let me go! '
# print(a*5) #在一行输出5遍
# print('helloworld'[2:]) #lloworld,同切片

# in 操作
# print(123 in [23,123,13]) #True
# print('elo'in'hello') #False

# a = '123'
# b = 'abc'
# c = a+b
# print(c) #123abc

#容器containers：列表（lists）、字典（dictionaries）、集合（sets）、元组（tuples）

#列表（lists）：相当于数组，长度可变、可包含不同类型的元素
# xs=[1,2,3]
# print(xs)  #输出[1, 2, 3]
# print(xs[2])  #输出3
# print(xs[-1]) #输出3，倒数第一个元素
# xs[2]='foo'
# print(xs)  #输出[1, 2, 'foo']
# xs.append('bar')
# xs.insert(2,'ins')
# print(xs) #1,2,ins,3
# print(xs)  #输出[1, 2, 'foo', 'bar']
# 删除有pop，remove，del
# x = xs.pop(1) # 删除并返回
# print(x)  #输出bar
# xs.remove('ins')
# del xs[0]
# del xs

#切片（slicing）一次性获取列表中元素,可以加第三个参数作为步长
# range(0,30,5)  #步长是5，即0,5,10,15,20,25
# nums = range(5)  #range(5)默认从0开始，0,1,2,3,4
# print(nums[2:4])  #2,3
# print(nums[2:])  #2,3,4
# print(nums[:])  #0,1,2,3,4
# print(nums[:-1])  #0,1,2,3

# list 其他操作
# x = [[1,2],1,1,[2,1,[1,2]]]# count计算某个元素在列表中出现的次数
# print(x.count(1)) #2

# a.extend(b) #一次性在列表a后追加b的内容 extend方法修改了被扩展的列表，原始的+操作会返回一个全新的列表

# lis = ['a','b','c','d','e']
# a = lis.index('b')  #获取b的下标 1
# print(a)

#循环
# animals = ['cat','dog','monkey']
# for animal in animals:
#     print(animal)  #分行输出cat、dog、monkey
# for idx,animal in enumerate(animals):  #使用内置enumerate函数在循环体内访问每个元素的指针
#     print('#%d:%s' % (idx+1,animal))  #分行输出#1:cat、#2:dog、#3:monkey

#列表推导
# nums=[0,1,2,3,4]
# square = []
# for x in nums:
#     square.append(x**2)  #计算平方值
# print(square)  #得到[0, 1, 4, 9, 16]
# square1 = [x ** 2 for x in nums]
# print(square1)  #得到[0, 1, 4, 9, 16]
# even_square = [x**2 for x in nums if x%2==0]
# print(even_square)  #输出[0, 4, 16]

#字典dictionaries：存储键值对，和map类似
#字典特点：无序，键唯一
# d={'person':2,'cat':4,'spider':8}
# for animal in d:
#     legs=d[animal]
#     print('A %s has %d legs' % (animal,legs))  #分行输出A person has 2 legs、A cat has 4 legs、A spider has 8 legs
# for animal,legs in d.iteritems():  #使用iteritems迭代
#     print('A %s has %d legs'%(animal,legs))

# dic2=dict((('name','alex'),))
# print(dic2) #{'name': 'alex'}

# dic3=dict([['name','alex'],])
# print(dic3) #{'name': 'alex'}

# dic3={'age': 18, 'name': 'alex', 'hobby': 'girl'}
# print(dic3['name']) #alex
# print(list(dic3.keys())) #['name', 'hobby', 'age']
# print(list(dic3.values())) #['alex', 'girl', 18]
# print(list(dic3.items())) #[('name', 'alex'), ('hobby', 'girl'), ('age', 18)]

# dic4 = {'name':'ttwen'}
# dic5={'1':'111','2':'222'}
# dic4.update(dic5) #5不变，4在前追加5的内容
# print('dic4:',dic4)
# print('dic5:',dic5)

# dic6 = {'name':'ttw','age':18}
# dic6.clear() #清空字典
# print(dic6)
# del dic6['name'] #删除字典中指定键值对
# print(dic6)

# dic7={'age':18, 'name':'ttw'}
# ret = dic7.pop('age') #删除指定键值对并返回
# a = dic7.popitem() #随机删除某组键值对，并以元组的方式返回

# dic8 = dict.fromkeys(['host1','host2','host3'],'test')
# print(dic8) #{'host2': 'test', 'host1': 'test', 'host3': 'test'}
# dic8['host2']='abc'
# print(dic8) #{'host1': 'test', 'host2': 'abc', 'host3': 'test'}

# dic6=dict.fromkeys(['host1','host2','host3'],['test1','tets2'])
# print(dic6)#{'host2': ['test1', 'tets2'], 'host3': ['test1', 'tets2'], 'host1': ['test1', 'tets2']}
# dic6['host2'][1]='test3'
# print(dic6)#{'host3': ['test1', 'test3'], 'host2': ['test1', 'test3'], 'host1': ['test1', 'test3']}

# dic5={'name': 'ttw', 'age': 18}
# for i in dic5:
#     print(i,dic5[i])
# for i,v in dic5.items():
#     print(i,v) #效果一样

#字典推导
# nums = [0,1,2,3,4]
# even_nums_to_square = {x: x**2 for x in nums if x%2==0}
# print(even_nums_to_square)  #输出{0: 0, 2: 4, 4: 16}

# 集合sets：独立不同个体的无序集合
# animals = {'cat', 'dog'}
# print('cat' in animals)   #True
# print('fish' in animals)  #False
# animals.add('fish')
# print('fish' in animals)  #True
# print(len(animals))       #3
# animals.add('cat')       #添加一个已经存在的元素添加不进去
# print(len(animals))       #3
# animals.remove('cat')    #移除
# print(len(animals))     #2

#循环loop：在集合中循环的语法和在列表中一样，但集合是无序的，不能做关于顺序的假设
# animals = {'cat', 'dog', 'fish'}
# for idx, animal in enumerate(animals):
#     print('#%d: %s' % (idx + 1, animal))##1: fish, #2: dog, #3: cat

#集合推导
# nums = {int(sqrt(x)) for x in range(30)}
# print(nums)  #{0, 1, 2, 3, 4, 5}

#元组tuple：是一个值的有序列表，可以在字典中用作键，可以作为集合的元素，列表不行
#元组一旦被初始化，不能改变内部元素
# d = {(x, x + 1): x for x in range(10)}
# # print(d)  #{(0, 1): 0, (1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4, (5, 6): 5, (6, 7): 6, (7, 8): 7, (8, 9): 8, (9, 10): 9}
# # t=(5,6)
# # print(type(t))  #<class 'tuple'>
# # print(d[t])  #5
# # print(d[(1,2)])   #1

#函数
# def sign(x):
#     if x> 0:
#         return 'positive'
#     elif x < 0:
#         return 'negtive'
#     else:
#         return 'zero'
#
# for x in[-1,0,2]:
#      print(sign(x))  #negtive,zero,positive
#
# def hello(name,loud=False):
#     if loud:
#         print('HELLO,%s' % name.upper())
#     else:
#         print('Hello,%s!' % name)
# hello('Bob')  #Hello,Bob!
# hello('Fred',loud=True)  #HELLO,FRED

#class类
# class Greeter(object):
#     #构造函数
#     def __init__(self,name):
#         self.name = name
#     #普通函数
#     def greet(self,loud=False):
#         if loud:
#             print('Hello,%s!' % self.name.upper())
#         else:
#             print('Hello,%s' % self.name)
#
# g = Greeter('Fred')
# g.greet()  #Hello,Fred
# g.greet(loud=True)  #Hello,FRED!

#Numpy:Python可科学计算核心库
# a=np.array([1,2,3])
# print(type(a))  #<class 'numpy.ndarray'>
# print(a.shape)  #(3,)
# print(a[0],a[1],a[2])  #1 2 3
# a[0] = 5
# print(a)  #[5 2 3]
# b = np.array([[1,2,3],[4,5,6]])
# 显示矩阵[[1 2 3]
#         [4 5 6]]
# print(b)
# print(b.shape)  #(2, 3),2行3列
# print(b[0,0],b[0,1],b[1,0])  #1 2 4

#Numpy创建数组的方法
# a = np.zeros((2,2))
# print(a)
# [[0. 0.]
#  [0. 0.]]

# b = np.ones((1,2))
# print(b)
# [[1. 1.]]

# c = np.full((2,2),7)
# print(c)
# [[7 7]
#  [7 7]]

# d = np.eye(2)
# print(d)
# [[1. 0.]
#  [0. 1.]]

# e = np.random.random((2,2))
# print(e)
# [[0.49179331 0.43272512]
#  [0.88331996 0.54478233]]

#Numpy访问数组
#切片
# a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print(a)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# print(a[0,1])  #2

# b = a[:2,1:3]  #获取前2行的1，2列
# print(b)
# [[2 3]
#  [6 7]]

# isMLGeek = True
# if isMLGeek:
#     print("True")
# else:
#     print("False") #python 中使用缩进区分代码块，在 c/java中使用{}区分

# 三引号不仅可以做注释，还可以像单引号双引号一样使用





















print("\nover!")