# 前言

目前开始转向了**PyTorch**，原因就不再赘述了。大势所趋呀，以前无论是实验和是生产全部是用**TensorFlow**。目前**PyTorch**几乎占据了整个学术界，所以为了很好的Follow别人的工作，学习**PyTorch**也是迫在眉睫的。
>我认为**PyTorch**主要是设计了四个比较精美的模块和类方别我们设计和调试模型，它们分别为提供模型组件的[torch.nn](https://pytorch.org/docs/stable/nn.html)，提供优化算法的[torch.optim](https://pytorch.org/docs/stable/optim.html)，提供封装数据的基类[Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)以及数据加载器[DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)。所以真正了解他们的工作，我们需要了解这四个模块，这里也分别针对这四个模块展开写.

## 目录
#### 1.数据准备

* 1.1 数据集的下载
* 1.2 处理并划分数据集

#### 2.制作自己数据集并加载
* 2.1 制作读取数据的txt文本
* 2.2 继承**DataSet**类
* 2.3 进行数据预处理
* 2.4 使用**DataLoader**加载数据

#### 3.设计模型
* 3.1 常用的组件
* 3.2 **forward**搭建模型
* 3.3 权重初始化

#### 4.选择损失函数和优化算法

* 4.1 **PyTorch**中常用的损失函数
* 4.2 **PyTorch**中常用的优化算法 

#### 5.训练和保存模型
* 5.1 训练模型
* 5.2 保存模型

---
### 1.数据准备
#### 1.1 数据集下载
这里我们自己从官网下载数据集，由于官网的数据是压缩格式的。所以这里要用**pickle**这个库来对其进行解压缩。这里就不用代码从官网上下载了，这里只用代码处理下载的压缩格式的文件。定义两个工具函数：

```
 #解压文件为python可以读取
def unpicle(file):
    with open(file, 'rb') as f:
        _dict = pickle.load(f, encoding='bytes')
    return _dict
# 创建文件夹的函数
def mkdirs(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)
```
然后我们分别再创建训练集合测试集的文件夹，注意这里官方的文件格式需要进行转换的。我们提取图片要转换成numpy格式的，具体使用下面的代码进行转换：

```
img = np.reshape(dataset[b'data'][i], (3, 32, 32))
img = img.transpose(1, 2, 0)#将通道放到后面
label = str(dataset[b'labels'][i])#标签转换
```
经过前面的转换后，我们就可以将numpy格式的图片进行保存了，这里保存如果使用**scipy.imsave()**这个方法时会报错，因为这个方法已经被丢弃了。具体解决方法可以参考这篇[文章](https://www.jianshu.com/p/22c74cd3707c)
#### 1.2 数据集划分
数据准备好后，我们还需要对获得数据进行划分，这是机器学习必须的流程。我们将数据划分训练集、验证集和测试集，具体这三个数据集的具体作用大家可以从网上很容易获得。训练集、验证集和测试集通常划分的比例为8:1:1的。我的划分数据集代码放在了**./utils/splite_datasets.py**这个脚本文件，具体实现写的也比较详细。这里需要强调的一点是我们在划分数据集的时候，将数据打乱是很重要的。这样避免了相同的数据分布在一起，所以这里实现添加了**random.shuffle(img_list)**这行代码。

### 2.制作自己数据集并加载
经过步骤一的操作，我们实现了将数据保存到硬盘。下一步就要将数据从硬盘加载到模型中来，这一块主要用到PyTorch中**Datasets**和**DataLoader**类。
#### 2.1 制作读取数据的txt文本
通常我们不是直接从硬盘中直接读取数据，有监督学习比较常见的做法是制作数据的方法是将一个样本在硬盘的路径和标签作为一个条目，具体的样式如下：
**/data/cifar-10-png/raw_train/__.png label**
这样的格式方便我们去读取数据，虽然PyTorch中提供了**ImageFolder**这个类可以直接提供图片的路径就可以，但是我还是提倡用这种方式，这是大家约定成俗的方式。实际实现起来也很方便，主要通过下面的代码：

```
def generate_txt(txt_path, data_path):
    for root, dirs, _ in os.walk(data_path, topdown=True):
        for dir in dirs:
            img_list = glob.glob(os.path.join(root, dir, "*.png"))
            lable_name = str(dir)
            for img in img_list:
                if not img.endswith("png"):
                    continue
                line = img + " " + lable_name + "\n"
                with open(txt_path, 'a+') as f:
                    f.write(line)
```

通过遍历图片的路径，然后将图片的地址和标签提取出来写入到文件中。我们这里分别制作了**train.txt,valid.txt,test.txt**这三个数据文件。
#### 2.2 继承**DataSet**类