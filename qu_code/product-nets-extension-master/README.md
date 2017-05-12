# product-nets
the extension of product-net (journal version)


## Dataset

Refer [APEX/Ads-RecSys-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets#ads-recsys-datasets) 
for feature engineering, and data access.

## How to Run

Configure environment and default settings in `__init__.py`.

Tune parameters in `main.py`.

Run `main.py`.

Tensorboard is integrated.

## log
firstly, LR = 77, FM = PNN2 = 79

### LR

### FM
FM 0.7785, factor=4, lambda=0.00002
FM 0.7909, factor=10, lambda=0.00002
FM 0.7695, factor=50, lambda=0.00002

### FFM
libffm 0.7959, logloss=0.54625, k=4, eta=0.2, lambda=0.00002, itr=15
libffm 0.7997, logloss=0.54198, k=4, eta=0.2, lambda=0.00002, itr=6

for循环与向量式的运行效率：
tensorflow-cpu batch=2000 rFFM
for elapsed : 0:04:30, ETA : 5:34:13 elapsed : 0:04:34, ETA : 5:39:11
vector elapsed : 0:03:47, ETA : 4:41:00 elapsed : 0:03:46, ETA : 4:39:45
batch=10000
for elapsed : 0:19:11, ETA : 4:29:36
vector elapsed : 0:19:28, ETA : 4:33:35
batch=1000
vector [  3.58819962e-06   5.00644207e-03   1.29657397e-01   2.84574032e-04]
elapsed : 0:04:21, ETA : 5:23:05
for [  3.95059586e-06   5.10080338e-03   1.68313227e-01   3.00381184e-04]
elapsed : 0:05:33, ETA : 6:52:13

在 tensorflow-cpu 上两种方法运行效率差不多 可能向量式略好
调大 batch 并不会降低训练时间 但是 train-auc 上升更快
在 batch 1000 上 vector 比 for 快了一个半小时 / 一轮

tensorflow-gpu
batch=2000
vector [  1.27792358e-06   3.11510563e-04   4.42347350e-01   2.60794163e-04]
elapsed : 0:07:21, ETA : 9:05:54
for [  1.66654587e-06   3.33654881e-04   1.58086450e-01   2.77166367e-04]
elapsed : 0:02:42, ETA : 3:20:32

batch=1000
vector [  1.50203705e-06   2.35679150e-04   2.02276549e-01   2.59990692e-04]
elapsed : 0:03:24, ETA : 8:28:27

for [  1.46389008e-06   2.33848095e-04   1.17071397e-01   2.58464813e-04]
elapsed : 0:02:01, ETA : 5:01:35

vector [  1.46627426e-06   2.31406689e-04   2.17327039e-01   2.57570744e-04]
elapsed : 0:03:36, ETA : 8:58:22

batch=10000
vector [  1.47342682e-06   1.50383711e-03   2.73534883e+00   2.97980309e-04]
elapsed : 0:38:10, ETA : 8:56:23

for [  1.64270401e-06   4.92475986e-03   1.12603605e+00   2.67875195e-04]
elapsed : 0:05:43, ETA : 4:41:08

在 FFM 中 embedding 层拆开与否不影响内存
for 循环与 向量式 占用内存也是一样的 但是向量式可以跑很大的 batch size 而且速度很快
embedding 放在 cpu 上极其慢 虽然能省一半显存

ffm k=4, decay=0.8/0.9
l2=1e-6 auc=0.7919 第一轮之后过拟合 每一轮的train会有涨幅 05-05-10:07:44
昨天跟小明讨论的结果是这种情况是不合理的 因为训练应该和看到的数据无关
l2=1e-5 auc=0.7814 保持稳定很差所以删掉了
按文件名 shuffle 发现抖动更加明显 0.7916 准备先实验一下加 noise shuffle_block 感觉没什么用 感觉还是要尝试全局shuffle
noisy=0.001 可能是加的noise太小了 训练/测试曲线和不加noise基本重合 test 最高 0.7918 05-07-06:12:13
乘法noise 0.1 0.7935 不过检查代码发现还做了norm 但是这个norm只加在w上 05-07-11:28:11
乘法noise 0.1 0.7919 没做 norm 基本重合说明 noise 0.1 没用？
乘法noise 0.1 对w v都做norm 发现实验做错了这个还没做 <----
没有noise 对w v都做norm 0.7971/0.5449 05-08-08:49:08
没有noise 对w v都做norm 0.7974/0.5446 05-08-15:08:05 batch=1000
0.3noise 做norm 05-09-01:56:47 0.7969
0.5noise 做norm 05-09-02:21:52 0.7964
0.1noise with norm 05-09-10:20:58 0.7971
0.05noise with norm 05-09-10:20:21 0.7971
说明乘性noise基本没用？

rm 05-08-13:04:09

### rFFM
rFFM k=4/10, l2=1e-6 与 不加l2 曲线基本重合, 在第一轮结束时达到 0.793 多一点 随后过拟合 04-29-11:04:59
但是 l2=2e-5 曲线瞬间变挫了好多, 最高只在一轮半的时候达到 0.7879, 感觉这个实验有点问题 重做一下
准备尝试 学习率decay 以及 更大的k？
k=50 跑不动把 batch 降到 100 目前看来效果变差

rffm k=4 l2=1e-6 auc=0.7934 过拟合的情况稍微好了一点 但是还是能看到每一轮的train会有涨幅 05-05-16:48:21
k=10 l2=1e-6 auc=0.7949 过拟合情况比 k=4 严重 每一轮的抖动更加明显 05-06-04:44:58
k=10 l2=1e-7 auc=0.7953 05-06-14:32:21 有过拟合的现象
k=4 l2=1e-6 norm=True 虽然曲线有所下降 但是貌似上升更快一点
k=4 l2=1e-7 norm=True auc=0.7935 过了十轮test也没有下降 可能还在缓慢上升 05-10-01:49:28
k=10 l2=1e-6 norm=True auc=0.7959 基本没有过拟合现象 test只是在几轮之后略微下降 train上也基本没有爬楼梯的现象 05-09-17:29:17
k=20 l2=1e-7 norm=True <---
可以调一下输入的 scale 不同模型未必需要做同样的 scale

### NetFFM

### FNN

### PNN1
pnn1 hidden 300 & 500 几乎重合 auc ~79.38

### PNN2
PNN2: 04-23-02:10:13
uniform -0.001, 0.001
learning_rate: 0.001, decay=0.95
dropout decreases AUC
todo: try batch_norm on hidden layers (not on embedding/output layers)

bn fails on pnn1 1 hidden, auc drops and loss increases
bn fails on pnn2 1 hidden, auc drops and loss increases
以上两个实验的问题在于 没有隐层 bn直接加在了输出层 以及 对bias也做了bn

pnn2 hidden 100 ~79.15 比较平滑
pnn2 hidden 100 batch_norm ~78.82 第一轮还没结束auc就开始下降 在第一个隐层后做了bn 不包含bias
pnn2 hidden 300 ~79.46 以下三条曲线几乎重合 每一轮结束auc会小幅下降 下一轮继续上升 第二轮结束时达到最高 04-24-10:42:16
pnn2 hidden 500 ~79.44
pnn2 hidden (300, 100) ~79.47 04-24-13:05:29
pnn2 hidden (300, 300) ~79.44 04-25-02:19:05
