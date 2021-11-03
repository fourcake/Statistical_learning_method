## 第三章 K邻近法

### 本章概要

1. k近邻法是⼀种基本**分类与回归**方法。分类时，给定⼀个训练数据集，对输⼊实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的多数属于某个类，就把该输⼊实例分为这个类。
2. k近邻法的三个基本要素：**k值的选择（常由交叉验证选择）、距离度量（常用欧氏距离，PL距离）及分类决策规则（常用多数表决）**
3. 
4. 

### K邻近模型

#### 距离度量

特征空间中两个实例点的距离是两个实例点相似程度的反映。k近邻模型的特征空间⼀般是n维实数向量空间，通常使用的距离是欧氏距离，也可用Lp距离(p≥1)，即：
$$
L_p(x_i,x_j)=(\sum_{l=1}^{N}|{x}^{(l)}_{i}-{x}^{(l)}_{j}|^p)^\frac{1}{p}
$$
当p=2时，即为欧式距离，p=1为曼哈顿距离。由不同的距离度量所确定的最近邻点是不同的。下图为p取不同值时，与原点距离为1的点的图形

<img src="images/%E7%AC%AC%E4%B8%89%E7%AB%A0%20K%E9%82%BB%E8%BF%91%E6%B3%95/image-20211103175134292.png" alt="image-20211103175134292" style="zoom:50%;" />

#### k值的选择

k值的选择会对k近邻法的结果产生很大影响，不可以很小，也不可以很大

k值的过小就意味着整体模型变得复杂，容易发生过拟合；k值过大，这时与输⼊实例较远的（不相似的）训练实例也会对预测起作用，使预测错误。在应用中，k值⼀般取⼀个比较小的数值。通常采用交叉验证法来选取最优的k值

#### 分类决策规则

一般使用多数表决原则，如果损失函数为0-1损失函数，对给定实例x，其最近邻的k个训[统计学习方法习题解答](https://datawhalechina.github.io/statistical-learning-method-solutions-manual/)练实例点构成集合$N_k(x)$。如果涵盖$N_k(x)$的区域的类别是$c_j$，误分类率为

<img src="images/%E7%AC%AC%E4%B8%89%E7%AB%A0%20K%E9%82%BB%E8%BF%91%E6%B3%95/image-20211103175203942.png" alt="image-20211103175203942" style="zoom:70%;" />

要使误分类率最小即经验风险最小，就要使分类中$\sum I(y_i=c_i)$最大，所以多数表决规则等价于经验风险最小化。

### 算法实现

#### 原始实现：

```python
import numpy as np
from numpy import *   
import operator
import matplotlib.pyplot as plt
%matplotlib inline
```

原始算法即为线性扫描，这时要计算输⼊实例与每⼀个训练实例的距离，算法如下：

```python
'''inX是用于分类的输入向量，dataSet为输入的训练样本集，labels为训练样本集对应的标签，k表取邻近的数目'''
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #计算欧式距离
    diffMat=tile(inX,(dataSetSize,1))-dataSet#inX复制成和dataSet一样多行的矩阵
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)#每一行相加
    distances=sqDistances**0.5#得到输入向量到训练样本集其它所有点的距离
    #对距离排序
    sortedDistIndicies=distances.argsort()
    #选取k个最近的点
    classCount={}#字典
    for i in range(k):
        voteIlabe=labels[sortedDistIndicies[i]]
        classCount[voteIlabe]=classCount.get(voteIlabe,0)+1#.get(voteIlabe,0):有voteIlabe对应值则返回，没有则为0
        
    #排序
    #operator.itemgetter(1)用法见百度，key整体功能为按字典（key，values）的values排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]#返回预测的类别
```

例：使用K-临近算法改进约会网站的配对效果（来源：机器学习实战）

步骤：

1. 准备数据：解析文本文件，处理行，列，建立数据矩阵（1000\*3）和对应的标签向量（1000\*1）

   ```python
   '''打开文件'''
   filename='./KNNdatingTestSet2.txt'
   fr=open(filename)
   '''读取行数，列数'''
   filelines=fr.readlines()
   numberOfrow=len(filelines)
   first_line = filelines[0]
   numberOfcolumn=0
   for i in first_line:
       if(i=='\t'):
           count+=1
   '''创建数据矩阵dataMat和标签向量classLabelVector'''
   dataMat=np.zeros((numberOfrow,numberOfcolumn))
   index=0
   classLabelVector=[]
   for line in filelines:
       line=line.strip()
       listFromLine=line.split('\t')
       dataMat[index,:]=listFromLine[0:3]
       classLabelVector.append(listFromLine[3])
       index=index+1
   ```

2. 分析数据：使用Matplotlib创建散点图观察

   ```python
   '''分析数据，可视化数据点分布'''
   fig=plt.figure(figsize=(10,4))#画布大小10*4
   ax=fig.add_subplot(121)#1*1的网格放一个图，若为221，则2*2的网格四个图里的第一个
   ax2=fig.add_subplot(122)
   classLabelOfFloat= array(classLabelVector).astype(float)
   ax.scatter(dataMat[:,1],dataMat[:,2],15.0*classLabelOfFloat,15.0*classLabelOfFloat)#后两个参数对应于?怎么加图例?
   ax2.scatter(dataMat[:,0],dataMat[:,1],15.0*classLabelOfFloat,15.0*classLabelOfFloat)
   plt.show()
   ```

   <img src="images/%E7%AC%AC%E4%B8%89%E7%AB%A0%20K%E9%82%BB%E8%BF%91%E6%B3%95/image-20211103175239904.png" alt="image-20211103175239904" style="zoom:80%;" />

3. 准备数据：数据归一化（处理不同范围的特征值取值），这里使用一般的归一化方式
   $$
   newValue=\frac{oldValue-min}{max-min}
   $$

   ```python
   minVals=dataMat.min(0)#返回每一列的最小最大值
   maxVals=dataMat.max(0)
   normDataSet=np.zeros(shape(dataMat))
   m=dataMat.shape[0]#1000
   ranges=maxVals-minVals
   normDataSet=dataMat-tile(minVals,(m,1))#将原矩阵纵向地复制m个，原矩阵1*3，复制完1000*3
   normDataSet=normDataSet/tile(ranges,(m,1))
   ```

4. 测试算法：我们可以采用错误率检测分类器的性能

   ```python
   hoRatio=0.10#测试比例
   numTestVecs=int(m*hoRatio)
   errorCount=0.0
   for i in range(numTestVecs):
       classifierResult=classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],classLabelVector[numTestVecs:m],3)
       print("the classifier came back with: %d,the real answer is: %d" % (int(classifierResult),classLabelOfFloat[i]))
       if(classifierResult!=classLabelVector[i]):
           errorCount+=1
   print("the classifier error rate is: %f"%(errorCount/float(numTestVecs)))
   ```

#### kd树实现

使用特殊的结构存储训练数据，以减少计算距离的次数

