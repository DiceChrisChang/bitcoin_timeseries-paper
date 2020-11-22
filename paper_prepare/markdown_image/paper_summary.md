[TOC]



## paper summary



1. 股价时间序列的分析与预测研究

> AR MA ARMA 的解释 —— 因素/变量 每个时间点数据/预测 
>
> p q 取值
>
> ARMA 的弱点 
>
> 线性/平稳性

### 步骤

* 检验平稳性
* 自相关/偏自相关判断 (初步估计 p,q 的值)
* 定阶 p,q (相关函数定阶法和 **AIC 准则**,**BIC准则**)
* 参数估计（最小二乘法）
* 显著性检验/随机性检验（残差序列满足白噪音）

### **平稳**

**差分使趋势平稳**

**严格平稳过程**（英语：Strict(ly) stationary process）或**强平稳过程**（英语：Strong(ly) stationary process）是一种特殊的随机过程，在其中任取一段期间或空间（{\displaystyle t=t_{1}-t_{k}}![t=t_{1}-t_{k}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4da7084c421111f9d409b728be605151f3118caa)）里的[联合概率分布](https://zh.wikipedia.org/w/index.php?title=%E8%81%AF%E5%90%88%E6%A9%9F%E7%8E%87%E5%88%86%E4%BD%88&action=edit&redlink=1)，与将这段期间任意**平移后**的新期间（{\displaystyle t=t_{1}+\tau -t_{k}+\tau }![t=t_{1}+\tau -t_{k}+\tau ](https://wikimedia.org/api/rest_v1/media/math/render/svg/9341b11a8e144bb219b978df8640ae5fc9730c6e)）之**联合概率分布相等**

white noise process 平稳

要求：

* 均值无明显变化
* 方差无明显变化

#### **平稳检验**

* ADF (置信区间)
* KPSS

#### **差分使平稳**

> First difference (d=1): y~t~ = Y~t~ - Y~t-1~
> Second difference (d=2): yt = (Y~t~ - Y~t-1~) - (Y~t-1~ - Y~t-2~)
> = Y~t~ - 2Y~t-1~ + Y~t-2~

*可以进行多次*

### Q：how to improve model ？

$$
y(查分) = y(t+1) - y(t)  /  t = time
$$

------



2. 基于选择性集成ARMA组合模型的零售业销量预测

> 过度拟合？

为了尽可能多的符合数据生成**变量很多**且**模型复杂**导致对**实际值预测不佳**

>线性与非线性组合<u>model</u>
>
>首先利用 ARMA 模型识别出商品销量数据中的线性成分，再对其残差中的非线性成分分别采用 SVR和ELM 方法进行预测 

###**选择性集成 ARMA 定阶方法定阶**

> 先通过 ARMA 模型识别出销量数据中的线性成分，然后采用不同的非线性回归预测方法对残差数据作出预测并将预测结果补偿给 ARMA模型

![arma_combine](/Users/zhangxinwan/where_paper/paper_prepare/markdown_image/arma_combine.png)模型特点：

* 遗传选取 p q 阶
* D维 ？ N - t 的时间 t 是如何出来的 ？

------



3. 基于时间序列模型的研究热点分析预测方法研究

​                                                                         *BUMMER*

------



4. 基于 ARMA 模型对股票“青岛海尔”成交量的分析预测 

#### **平稳方法**

1. 低阶差分
2. 季节差分
3. 对数差分
4. n 阶差分

**白噪声检验一般采用检验统计量 Q 的方法**

### 模型的有效性检验

> 模型拟合的效果主要通过检验残差项是否为白噪声序列进行评价. 当残差项为白噪声序列时，说明差项中不包含相关信息，时间序列中的信息被完全提取，此时模型拟合效果显著

------



5. 金融危机下人民币汇率的ARMA模型预期

### 滞后项

滞后变量就是从时间上看比当期变量滞后的变量。在[计量经济模型](https://wiki.mbalib.com/wiki/%E8%AE%A1%E9%87%8F%E7%BB%8F%E6%B5%8E%E6%A8%A1%E5%9E%8B)中，有时需要用解释变量或被解释变量的滞后变量做解释变量。例如给出宏观消费模型如下。

　　*Y**t*=α0+α1*Y**t* − 1+β1*X**t*+β2*X**t* − 1+μ*t*

其中*Y**t*表示宏观消费，*X**t*表示宏观收入。*Y**t* − 1和*X**t* − 1分别表示前一期的消费和收入。*Y**t* − 1和*X**t* − 1称作一阶滞后变量。当回归模型中出现m阶滞后变量时，估计模型参数的[样本容量](https://wiki.mbalib.com/wiki/%E6%A0%B7%E6%9C%AC%E5%AE%B9%E9%87%8F)减少为T-m。

**静态模拟**与**动态模拟**

6. 基于时间序列的股票价格分析研究与应用

### 价格受影响因素

* 国家政策
* 投资心理
* 发展状况
* 汇率变化
* 利率水平

### GRACH

如果担心residual of ARMA exhibit heteroscedasticity, 那么看看sqared ACF 是不是像white noise, 结论是是否的话，用GARCH模型。

b) ARIMA/SARIMA
参数怎么选见其他回答。。同理，最后得到的recovered residual 应该是white noise.. 如果怀疑 heteroscedasticity. 参见GARCH model

## python ARMA模型建立

1. 导入库及准备

``` python
from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IndexData = DataAPI.MktIdxdGet(indexID=u"",ticker=u"000001",beginDate=u"20130101",endDate=u"20140801",field=u"tradeDate,closeIndex,CHGPct",pandas="1") 
IndexData = IndexData.set_index(IndexData['tradeDate'])
data = np.array(IndexData['CHGPct']) # 上证指数日涨跌
IndexData['CHGPct'].plot(figsize=(15,5))
```

[举例说明](https://uqer.io/v3/community/share/57988677228e5ba28e05faff)

标题设计

(# ## ###)

​	

# 理论假设

## 	时间序列

### 		原理

### 		应用假设	

## 	ARMA/ARIMA



# 建模过程数据处理

## 	建模过程

## 	获取数据

### 			获取

### 			整理

## 	初步描述处理



# 应用结果与分析
## 预测能力



# 结论总结
