
## Optimizer
  **Second-order optimization methods:**
  * Newton's method:
    $X^*=X_0-H(f)(X_0)^{-1}\nabla_xf(x_0)$
    如果函数在某个部分是接近于二次的，那么迭代得更新权重可以很快的跳到局部最优点，比一般的梯度下降快。
  * 这个函数在深度学习中是非常复杂的。
  * condition number:
    $max|{\lambda_i\over\lambda_j}|$
    Hessian 矩阵的最大特征值和最小特征值的比例越小越好。
    如果condition number 很大，那么梯度下降算法表现不是很好。那是因为在某个方向，偏微分下降很快。但在另一个方向，偏微分下降得很慢。所以Hessian矩阵可以预测哪个方向下降的最快，但这并不是最希望的方向。
  * 由泰勒二次展开得到的：
    $f(x)=f(x^k)+f'(x^k)(x-x^k)+{1\over2}f''(x^k)(x-x^k)^2$
    对$f(x)$求导后得到
    $x=x^k-{f'(x^k)\over{f''(x^k)}}$
    其实本质上是找到$f'(x)=0$的点，但因为泰勒展开需要满足x趋近$x^k$,所以在非二次函数中，直接用牛顿法是不能直接找到optimal solution的，但如果初始点很接近最优点的话，会很快找到。


**Momentum:**
* 损失函数的曲线可能有近似平地的区域，在这些地方gradient很小，所以用一般的梯度下降法很难收敛。
* SGD+Momentum:
  $V_t=rV_{t-1}+(1-r)g_t$
  $X_t=X_{t-1}-V_t$
  $V_t=rV_{t-1}+(1-r)g_t$
  $\quad=r[rV_{t-2}+(1-r)g_{t-1}]+(1-r)g_t$
  $\quad=r^tv+(1-r)\sum_{i=0}^{t-1}r^ig_{t-i}$
  $if \quad i={1\over{1-r}} \quad r^{1\over{1-r}}=(1+r-1)^{1\over{1-r}}=e^{-1}\quad r^i->0$
  所以如果r足够小的话，当$i={1\over1-r} $时，后面的部分会极小可以忽略不计，所以相当于包含了${1\over1-r}$步的梯度信息。
* SGD 在某个梯度明显比其他梯度更陡的时候效果不好。因为这样会在在某个梯度下降很快，但明没有在理想的方向下降的很快。
* $V_{t-1}$相当于是包含了前面许多步的梯度信息，momentum可以削弱了那些梯度经常变化的方向(目前的g）带来的影响，而更多考虑以前梯度的信息。这和牛顿法有一样的思想。
  

**Higher-order nonlinearities:**
* 二阶思想和动量思想都假设在minimum是quadratic形式的。增加在某个方向梯度很小的step size，而减小在某个方向梯度很大的step size。
* 在训练deep model时，高阶会带来很多的非线性，不像二阶那样对称。
  

**AdaGrad:**
* $S^{t}[u-v]=s^{t-1}[u-v]+(g^{t}[u-v])^2$
  $W^{t+1}[u-v]=W^t-{\alpha_t\over\sqrt{s^t[u-v]}+\epsilon}g^{t}[u-v]$
* 相当于随着权重的更新，学习率也在更新。
* 但很少用
  

**RMSprop=Adagrad+forgetting(moving window):**
* $S^{t}[u-v]=\sigma s^{t-1}[u-v]+(1-\sigma)(g^{t}[u-v])^2$
  $W^{t+1}[u-v]=W^t-{\alpha_t\over\sqrt{s^t[u-v]}+\epsilon}g^{t}[u-v]$
* 相较于Adagrad，多了移动之后，更关注于目前的梯度值带来的影响，如果目前梯度也很小的话，步长不应该太小。而不是像Adagrad一样随着权重更新，步长不断变小。

**Adam=RMSprop+Momentum:**
* $m^{t}[u-v]=\beta_1 m^{t-1}[u-v]+(1-\beta_1)(g^{t}[u-v])$
  $S^{t}[u-v]=\beta_2 s^{t-1}[u-v]+(1-\beta_2)(g^{t}[u-v])^2$
  $W^{t+1}[u-v]=W^t-{\alpha_t\over\sqrt{s^t[u-v]}+\epsilon}m^{t}[u-v]$
* 加入momentum是为了找到合适的方向，而adagrad是为了找到合适的步长
* 最经常用的优化算法，收敛速度也是最快的。

