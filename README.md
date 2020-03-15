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
我们将图片划分为训练集、验证集合测试集，接下来就是让PyTorch能读取这批数据。这里就需要按照PyTorch读取图片的机制和流程来进行代码的编写了。
##### Dataset类
要能让PyTorch读取我们的图片，主要是通过**Dataset**类,Dataset类作为所有的datasets的基类存在，所有的datasets都要继承它，这一点类似于C++中的虚基类。我们可以看下源码实现细节：

```python
class Dataset(object):
"""An abstract class representing a Dataset.
All other datasets should subclass it. All subclasses should override ``__len__``, 
that provides the size of the dataset, and ``__getitem__``, 
supporting integer indexing in range from 0 to len(self) exclusive.
"""
	def __getitem__(self, index): 
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError
	def __add__(self, other):
		return ConcatDataset([self, other])
```
从上面的源码可以看出，Dataset的子类必须实现```__getitem__```和```__len__```，否则程序将报错！这里重点看**getitem**函数，它接受一个index即单个图片的索引，然后返回图片数据和标签，这个index通常指的是一个list的index，这个list的每个元素就包含了图片数据的路径和标签信息。**__len__**就不用多说了，就是数据集中样本的总数。

这里需要重点看的是getitem函数，getitem接受一个index，然后返回图片数据和标签，这个index通常指的是一个list的index，那么这个list的每个元素就包含了图片数据的路径和标签信息。归纳一下就是如下的3个基本流程：

>**1.制作存储了图片的路径和标签信息的txt**

>**2.将这些信息转化成list，该list每一个元素对应一个样本**

>**3.通过getitem函数，读取数据和标签，并返回数据和标签**

实际操作中，我们只要通过DataLoader就可以获取一个batch的数据，其实触发去读取图片操作的是Dataloader里的__iter__(self)，因此这里如果让PyTorch读取自己的数据集，这里归纳一下需要两步：

> **1. 制作图片数据的索引**
> **2.构建Dataset的子类**

##### 制作图片数据的索引

这个非常的简单，就是读取图片路径，标签，保存到txt文件中。只要按照一行一条条目就可以了，这里需要强调一点的是在这txt文本中的图片路径是和训练脚本指定的相对路径，如果训练脚本位置改变该条目路径也要相应的改变，所以我的习惯是放绝对路径。所以这一点要注意的！这一段代码可以参考```./utils/generate_txt.py```这个脚本文件。

##### 继承Dataset父类

这一步很关键，这一步就是建立起与**Dataloader**通信的关键一步，这里就像上文提到的关键是要重写它的**__getitem__**和**__len__**方法，具体代码如下：

```python

"""
构建自己的数据集类，需要实现Datasets这个类。需要关键实现__getitem__和__len__方法
1.制作图片数据的索引
2.构建Dataset子类
"""
from PIL import Image
from torch.utils.data import Dataset


class MyDatasets(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        imgs = []#定义图片数据的list
        with open(txt_path, 'r') as f:
            fh = f.readlines()
            for line in fh:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))#img单个list添加图片路径和标签
        self.imgs = imgs
        self.transform = transform  # tansform图片变换
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert("RGB")  # 这里的图片是一个PIL对象
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
        
```

这段代码首先进行初始化，初始化中从准备好的txt里获取图片的路径和标签，并且存储在self.imgs，就是txt中的一行。个人习惯也将图片的预处理transform放在初始化中，transform是一个Compose类型，里边是一个list，list中会定义各种对图像进行处理的操作，该操作共有22个预处理操作，这里就不再赘述了。

我们着重看看核心的**getitem**函数：

首先我们从初始化中根据索引取出imgs的一个条目，这样就取出了一个包含图片路径和图片标签的条目。下面就要主要强调一点的是**DataLoader**中加载的是PIL对象，所以这里要将图片打开，就是通过```Image.open(fn).convert("RGB")```打开即可，下面就是对图像进行transform预处理了。这样我们自己的MyDataset就建立好了，下一步就是交给**DataLoader**进行加载了！
#### 2.3 进行数据预处理

#### 2.4 使用**DataLoader**加载数据
从上节中我们实现了构建自己的Dataset子类，并实现了两个重要的方法**getitem**和**len**方法，其实它实现的一个重要作用就是获取图片的索引以及定义如何通过索引读取图片及其标签。但是这只是第一步，这样数据并没有到模型中，要触发MyDataset去读取图片及其标签却是在数据加载器**DataLoader**中。
一句话概括就是：**从MyDataset来，到MyDataset去**。在我们实现的MyDataset类中，在该实例中有路径，有读取图片的方法，然后需要PyTorch的一系列规范化流程，才会调用MyDataset中的**getitem**函数，最终通过**Image.open()**读取图片数据。然后对原始数据进行一系列预处理，将数据转换为Variable类型，最终成为模型的输入。

**详细流程如下：**

1.从MyDataset类中初始化txt，txt中有图片路径和标签

2.初始化**DataLoader**时，将train_data传入，从而使**DataLoader**拥有图片的路径

3.在一个iteration进行时，才读取一个batch的图片数据enumerate()函数会返回可迭代数据的一个"元素"

4.**class DataLoader()**再调用**class _DataLoaderIter()**

5.在**_DataLoaderIter()**类中会跳到__next__(self)函数，在该函数中会通过```indices=next(self.sample_iter)```获取一个batch的indices再通过```batch = self.collate_fn([self.dataset[i] for i in indices])```获取一个batch的数据，在```batch = self.collate_fn([self.dataset[i] for i in indices])```中会调用```self.collate_fn```函数

6.**self.collate_fn**中会调用MyDataset类中的__getitem__()函数，在__getitem__()中通过```Image.open(fn).convert("RGB")```读取图片

7.通过Image.open(fn).convert('RGB')读取图片之后，会对图片进行预处理等一系列transform后，最后返回img,label,再通过self.collate_fn来拼接成一个batch。一个batch是一个list，有两个元素，第一个元素是图片数据、第二个是标签数据。

8.将图片数据加载到模型中