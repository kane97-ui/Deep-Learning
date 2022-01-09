## CNN
**computation of size of the output:**

${W^{'}}={(W-F+2P)\over S}+1$
F:size of kernel
W:input size
p: size of padding

**全联接神经网络的问题：**
* output的每一个神经元都与上一层的输入神经元连接
* 随着输入图像的尺寸的增加，权重会呈指数增加
* 远距离像素点之间有很低的相关性
  

**局部连接神经网络：**
* 稀疏连接：隐藏层的神经元只与部分的区域相连
* 受生物系统的激发，一个细胞只对某个很小的区域敏感，这叫感知区域。
* 在更高层的节点会对input有更大的感知区域
* 如果一直像这样一层一层的网上搭，会越来越global
  

**共享权重：**
* 翻译不变性：抓住局部区域的统计意义，并且他们在位置上是独立的。
  * 某一类权重或者某一个卷积核是对某一类特性的解析和捕捉，一个卷积核在图像上的一移动就是为了捕捉不同的区域的某以个特性。
  * 在低层是为了捕捉某些细节特性，而越往高层，特性更加整体性。
* 不同位置的隐藏层的节点享用同样的权重。这样会大大降低权重数量。
* 在某些应用中，我们可能知识局部的共享权重或者在顶层不共享权重。
  

**Zero-padding：**
* zero-padding使输入更加的宽
* 没有zero-padding，表征的宽度会被一层一层的稀释
* 为了避免网络快速的稀释，可以用更小的filters
* 通过zero-padding，阻止了表征随着深度的加深而稀释。
  

**Normalization:**
* covariaten shift: 一个samples在经过函数时，它的均值和方差不会变，但是形状不会怎么变。
* 可以避免梯度消失和梯度爆炸：如果每一层的输入样本的分布不是均值为0，方差为1的话，随着covariate shift，均值和方差可能呈指数型变大（节点的值变大），这时用backward算梯度时：可能会梯度爆炸。当节点值太大时，也有可能造成激活函数梯度极小，随着backward的进行：梯度会呈指数型减小，从而梯度消失。
* 如果一个输入的某个特性的feature范围明显高于其他特性，那么在训练的时候，那么网络会主要调整该特性的权重而忽视了其他特性。
* 输入的变量应该是不相关的：
  * 如果输入时不相关的，那么很有可能在解决一个权重的时候不用考虑其他权重。
  * 如果输入是相关的，我们必须同时解决不同的权重，是非常困难的问题。
  * PCA技术可以用来去除输入的线性相关性。
* 减小梯度对初始值的依赖。
* 在使用higher learning rates时，减小了divergence的风险。
* 在使用saturating nonlinearities的激活函数时，防止网络陷入饱和状态(造成梯度消失)。

**Initialization：**
* 不能把权重全初始化为0:梯度始终是0当BP时
* 随即初始化： e.g. Gaussian with mean zero,N(0,001)
* Xavier initialization:
  * 如果连接节点v的权重有n个，那么权重需要～$N(0,{\sigma^2\over n})$，这样节点v就恰好$～N(0,{\sigma^2})$
  * 这里假定父节点均值为0
  * 需要激活函数的均值也为0:对于ReLUs不满足
* Kaiming initialization:
  * 主要针对ReLUs
  * 因为RuLus激活函数只会激活一半的神经元，所以只有一半的权重（n/2）连接了节点V，所以权重的初始化$~N(0,2\sigma^2/n)$


**Type of Normalization**
* Batch Normalization:同一个batch中不同sample之间的同一个像素点（同一个channel）的normalization
* layer Normalization:同一个sample的不同通道同一个位置之间的normalization
* Instance Normalization: 同一个sample的同一通道不同位置（不同像素点）的normalization
* Group Normalization:
  

**Pooling：**
* 分为max-pooling和average-pooling
* 非线性的下采样
* 输出和输入的feature map的数量是一样的，只不过像素点数量减少
* 为上面的神经层减少了计算量，但提供了解释不同性的形式
* Pooling非常有用，如果我们在乎某种特性是否存在，而不是这个特性在哪个位置
* 不需要下采样的pooling（s=1)
* 下采样的pooling更经常使用
* 顺序：Conv->Normalization->Pooling

**激活函数的选择：**
相较于tanh和sigmoid的，更倾向于选择relu：
* sigmoid和tanh都是指数函数，计算相较于relu更加复杂
* 在深度神经网络中，如果sigmoid或者tanh激活函数刚好在饱和区，那么通过向后传播计算梯度的时候，很容易造成梯度消失，收敛起来会非常慢
* relu在部分区域是0，这可以让部分神经元不被激活，让网络更加稀疏，可以减少参数之间的相关性，也可以避免过拟合。
  

**warm start/pre-training:**
* 如果想继续训练更多epoch：
  * 保存权重，然后再次开始训练。
  * 需要谨慎地修改lr
* 可以用其他训练好的任务作为初始点，然后fine tuning一个新的任务。
* 