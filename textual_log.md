# Textual Analysis
26号的一个任务是取数据，下一步怎么走，模型？做一个对大家都有用的东西，长期好使的。今天Kevin说的很对的，你要用他们这些资源，不要只是坐在那里。

https://docs.google.com/document/d/1nT0ehbe-AxpMOiu0E61ydnKUcmr3yEfHlG-jg-I-Ui8/edit?ts=5dfbebf8

 First due on 12/26
 Describe the details about the model in the next step

通过google drive-scraping code-finaldatascraping.ipynb文件中的代码，下载分配到的股票10k 和 10q。 每人分配2个sector，每个sector有2014-2018共5年数据需要下载，每一年有10家代表公司，这十家公司的10k，10q都需要下载。每人需要得到2(sector)5(year)10(company) = 数量为100 的数据，每个数据都是txt格式。请注意年份和对应的公司，不要下错数据！
7天时间，12/26日前汇总数据。

# 12/24
只要将ticker读入Python，更改第四个box的tickercikdf内容到相应的格式即可。在第四个box之前加一个框：

ComputeCosineSimilarity函数，计算两个词的相似度。毫无意义。。。

Class3的applyingregexes10kssolution非常重要，重点介绍了re正则库。
- \s是匹配空格
- \s[A-Za-z]{2}\sEquity
- https://juejin.im/post/5b5db5b8e51d4519155720d2
- https://blog.csdn.net/fanzhen_hua/article/details/2050015

不同的年份，如果直接跑循环，这个程序会默认判断相应的ticker已经取得。不对！这个程序是按照股票ticker抓取每只股票的相同年数的财报。

把程序改成每次只抓一年。

# 12/26
*搭建本地sql服务器，实现链接Python

搭建数据库 https://www.cnblogs.com/opsprobe/p/9126864.html

apt install mysql-server
netstat -tap | grep mysql

mysqlsecureinstallation

核心问题就是mysql密码设置不成功造成的
mysql> update mysql.user SET authenticationstring=PASSWORD('qwertyuiop098'), plugin='mysqlnative_password' WHERE user='root'

windows实现远程访问

现在配置mysql允许远程访问，首先编辑 /etc/mysql/mysql.conf.d/mysqld.cnf 配置文件
vim /etc/mysql/mysql.conf.d/mysqld.cnf
注释掉bind-address = 127.0.0.1

保存退出，然后进入mysql数据库，执行授权命令：
mysql -u root -p

mysql> grant all on *.* to root@'%' identified by '你的密码' with grant option;

mysql> flush privileges;    # 刷新权限

mysql> exit`
然后执行exit命令退出mysql服务，再执行如下命令重启mysql：
systemctl restart mysql

# 12/27
创建用户和授权
CREATE USER 'pig'@'%' IDENTIFIED BY '123456';
grant all on . to root@'%' identified by '你的密码' with grant option;

sudo ufw status verbose

确定project步骤
数据预处理，分词，去停词，steming，tf-idf特征提取
主题模型的实现
主题和目标公司之间的匹配，选择特征化方式使得主题和股票之间可比
构建etf回测
主题模型
语料

主题分布、离散的

主题模型可以一定程度上解决一次多义和多词一义的问题，因为其具有的映射模式，一个主题可以有很多不同的词，一个词可以属于不同的主题。

tf-idf特征提取， lda主题模型，主题强度

词袋模型假定一篇文章中又若干词语组成，词和词之间独立，无顺序，只是大集合；

每一个主题看成一个随机向量，

gensim库

# 12/29
从python写入sql向量化之后的数据，每个documents存为一个向量。
https://blog.csdn.net/Mr__lqy/article/details/85719603
https://blog.csdn.net/zhanshirj/article/details/74732718

gpu加速计算
https://lightgbm.readthedocs.io/en/latest/

RAPIDS

关于我们的etf如何构建我现在有三个思路：

General Assumptions
• ETF = W1T1 + W2T2 + W3T3 + ... + WkTk （主题模型对我们对目标公司得到k个主题Tk，每个主题前面的Wk作为我们投资组合的权重，有sum(Wk)=1），因此我们的目标变成寻找每个主题对应在市场里的投资组合如何构建。
• 每个公司的10k在经过tf-idf特征化处理之后，都会得到一个N纬的向量，对应N个词，这N个词来自整个语料库（所有公司的10K组成），我们称为词基，记做B。
• 由LDA主题模型所生成的每个主题T(i)是一个N维的词分布，也是一个N纬的向量，对应的实际含义也是N个词，唯一的区别就是这个向量所有元素加起来等于1，原因是它是一个分布。

机器学习classifier
这个是我觉得相对来说最简单的一个想法，比较省事。

1.1 Special Assumptions

• 每个主题T(i)只对应一个行业（关于实际一个主题到底包含几个行业信息我们可以从主题模型输出的主题强度来看）
• 按照从Topic -> 行业 -> 代表行业的投资组合，来构建etf
• 直接寻找行业ETF作为构建该行业的投资组合

1.2 模型描述

如果我们回顾Gerneral Assumptions能发现如下事实：每一篇10k特征化之后得到的向量和主题模型中得到的每个主题，唯一区别就是主题模型向量是一个分布，也即sum（所有元素）=1. 因此，当我们对每篇10k得到的结果都统一做标准化之后，以上两者之间就可进行比较了。

因为我们已经取得12个行业代表公司的历年10k，经过特征提取之后我们假设得到Z个向量，由于行业已知，因此我们可以方便的对这Z个向量贴标签（1-12）。如此这般，这Z个向量构成了机器学习的输入数据，任务为分类任务。以单个公司的数据做模型训练，得到训练好的模型M1. 这里的模型可以有很多种选择（SVM，朴素贝叶斯，决策树xgboost，KNN），【这里在第一遍实验跑代码的时候关注一下速度，如果处理时间过长就要考虑放弃sklearn库，转向RAPIDS库用gpu加速计算】，因为这些模型的输入都很一致，所以我们可以方便的都做看看哪个效果好就用哪个。

对M1，我们输入主题模型得到个各个主题向量，由于之前的标准化处理使得主题和单个公司之间可比，因此我们能做到对所得到的主题进行行业分类，进而得到ETF。

聚类算法（k-mean聚类）
2.1 Special Assumptions

• 每个主题含有不止一个行业的信息
• 跳过寻找行业直接topic -> 投资组合构建
• 我们选取的代行业代表股票作为基础选股池子

2.2 模型描述

同样需要通过对10k标准化使得公司和主题之间可比。自此，将K个主题向量，和所有10k之间进行聚类，通过选取特定的k（使得模型最终聚成几类），理想的情况是我们选择k=K，也即是主题模型得到的主题个数，然后每个主题都不在同一类，但是每个主题所在类里都有一些公司。

eg第一类：包含Topic1，公司A201710k，A201610k, B201710k, C201710k; 那么对于topic1我们形成的投资组合就是（0.5A+0.25B+0.25C）。进一步，第一个topic在我们总的etf中最后得到投资组合就是W1（0.5A+0.25B+0.25*C）

线性模型
3.1 Special Assumptions

• 所有的topic都是我们选取的目标公司的线性组合
• 通过解一大堆线性方程组来得到投资组合

3.2 模型描述

不想描述了。解方程步骤太多，感觉不太实际。好处就是，这个模型是完全线性的，可解释性更强。

## 标签提取

取business和risk factor两个标签，如果没有则不取；
方法：通过b标签定位，选取itme1和item3之间的内容，如果这部分中包含BUSINESS和Risk Factor字样的标签，则保留提取结束；若只包含BUSINESS则只保留itme1；else这篇10k不提取任何数据。

# 1/1

- 完成标签提取任务（参考Andy给的sample）
手头现在有的这些老师、教授的资源大概率是好过你在网上去找去学的方式的，好好利用手头有的东西。不要总干早车轮的事情。
&nbsp;是JavaScript中的空格占位符。

andy在拿到爬去的html的数据之后先做了一步soup中tag的清洗，我们效仿。看了Andy的代码以后发现，其实Andy在这一步已经做了详细的代码（就是包括下载，取出business session之类的），但是它速度巨慢。一个公司5年的数据大概可能10min左右？所以我觉得我们还是保留之前我们爬取的数据，然后借鉴Andy的数据处理部分就ok。

enumerate函数是干嘛的？

Andy提供的section_extract函数不是很通用，对于有些10k的business和risk factors不在item1a，可能直接就是item，所以我现在考虑把范围直接拓宽到item1.

# 1/4

- 解决LDA模型的输入问题，紧迫！决定了我们当前的tf-idf特征化处理是否需要调整。
    + https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    + 
- 解决10k-10q section抽取问题，以及不能找到相应section的问题


# 1/8

- 对数据改名字

A.
现在的问题是10k长的样子实在是太不一样了，怎么能统一的拽出来要的business和riskfactor呢
？
把extract函数单独抽出来，对单个10k操作；
然后进一步考虑extract函数怎么改进搜索条件；
然后path在dfidf里进行操作，锁定path对那个目录下进行特征化；



B.
LDA要求的输入是长什么样子的？能不能和现在已有的工作对上？

lda实现过程里：bow_corpus(来自word of bags模型) -> 


# 1/11

一个新的句子->bow->LDA

- [X] 机器学习模型数据准备已完成

# 1/12

Meeting
1. 当前工作进度，go through notebook

2. 剩下待完成工作
    - 每个人改名字（今天之内完成），今晚下下来数据开始训练模型
        + 按照patti之前规定的格式，代码已有
        + 最后每个人汇总成一个行业一个文件夹

    - 机器学习模型训练，数据准备工作已经完成
    - extract函数改进
    - write up

    - 回测预测2018年任务
        + 收集整理目标公司10k数据
        + 收集目标公司trading data
        + 收集用来代表行业ETF的可交易证券信息

3. 任务分工

- Read goole docs!!!!!!!

- Hongfei：
    + 机器学习模型训练，数据准备工作已经完成 shuwen
    + extract函数改进 huan

- Xuhuan, Patti, Shawn
    - 回测预测2018年任务
        + 收集整理目标公司10k数据
        + 收集目标公司trading data
        + 收集用来代表行业ETF的可交易证券信息


去停词部分去除数字，a\an\the 这种冠词




# 1/13

Xuhuan:
1. 只保留business部分

# 1/25
要做ppt到讲演部分，基本告诉大家我们做了什么东西怎么做的，主要问题是什么，解决。
