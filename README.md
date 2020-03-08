#前言
目前开始转向了**PyTorch**，原因就不再赘述了。大势所趋呀，以前无论是实验和是生产全部是用**TensorFlow**。目前**PyTorch**几乎占据了整个学术界，所以为了很好的Follow别人的工作，学习**PyTorch**也是迫在眉睫的。
>我认为**PyTorch**主要是设计了四个比较精美的模块和类方别我们设计和调试模型，它们分别为提供模型组件的[torch.nn](https://pytorch.org/docs/stable/nn.html)，提供优化算法的[torch.optim](https://pytorch.org/docs/stable/optim.html)，提供封装数据的基类[Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)以及数据加载器[DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)。所以真正了解他们的工作，我们需要了解这四个模块，这里也分别针对这四个模块展开写.

##目录
#### 1.数据准备

* 数据集的下载
* 处理并划分数据集

#### 2.制作自己数据集并加载
* 继承**DataSet**类
* 进行数据预处理
* 使用**DataLoader**加载数据

#### 3.设计模型
* 常用的组件
* **forward**搭建模型
* 权重初始化

#### 4.选择损失函数和优化算法

* **PyTorch**中常用的损失函数
* **PyTorch**中常用的优化算法 

#### 5.训练和保存模型
* 训练模型
* 保存模型