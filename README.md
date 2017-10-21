# DCASE2016
声音场景识别
DCASE2016
- 任务介绍以及数据库下载：http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification
- 依赖库：numpy、scipy、librosa、keras、tensorflow or theano
- 实验采用Mel能量谱+CNN+随机森林
- 试验的结果：
- ![image](https://github.com/DeepLJH0001/DCASE2016/blob/master/images/%E9%94%99%E5%88%86%E7%9F%A9%E9%98%B5.png)
- cnn本身的feature map特征其实是非常稀疏的、即使采用过拟合手段有dropout，交叉验证、早停法、权重衰减、正则化，仍然有一些数值较低的权重，而不是0。
- 在有噪声的情况下（场景声音混杂了其他声音事件、如交谈声、风声等），很多神经元节点其实本身的权重都不会偏向0，而是以一个较小的值存在、本实验思想主要尝试使用随机森林的自助重采样，直接摒弃部分CNN的冗余特征构建决策树、bagging方式实现声音场景的识别，而未被采样到的袋外数据可以做validation data。
