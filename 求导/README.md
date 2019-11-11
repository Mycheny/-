# 动机

作者 Yangtf

最近一直在求各种导数，于是就想写一个自动求导的算法。 其实python中的theano就有这个功能，但想了想，思路不难，于是就动手实现了一个。

本来想用c++实现了，但发现c++写各种问题，内存管理、操作符重载都不尽人意。花费了不少时间后，决定换语言。 Java是第一熟练语言，但不支持操作符重载，奈何？ 于是转战python。 

# 源代码路径

最新的源代码在这里。
http://git.oschina.net/yangtf/python_exp


# 思路

##函数的表示

将函数表达式表示为一个表达式树。

![这里写图片描述](http://img.blog.csdn.net/20170608224650693?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGFpSmkxOTg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

那个这个表达式树如何构建呢？　要自己写语法分析么？　太麻烦，有种比较简单的办法，就是使用操作符重载来实现。

定义一个类Ｅ，重载它的 + - * /  **（乘方）操作，在重载中，进行二叉树的构建。

## 节点类型
在这个表达式树中，主要应有三种节点类型。
其一，常数节点。如 2，3 
其二，变量节点，如 a,b,x,y之类。
其三，操作节点。如 + , - ,* , / ,乘方等。


## 求导方法

有了表达式构成的二叉树，下面就是求导了。

对常数节点求导，结果为0 。
对变量节点求导，有两种情况。如
$$ f(a,b) = a^2 + 3b $$

这个函数对$a$ 求偏导，那么就将b节点看成是一个常数,求导结果为0。
对于保存了a的节点，求导结果为1。
 


求导的方法就是那些求导公式，举例：

$$ (x+y)' = x' + y' $$

求导看这篇文章 http://blog.csdn.net/taiji1985/article/details/72857554

上面的公式，对于一个根为‘+’的二叉树，分别对其左子树和 右子树进行求导，然后将求导得到的和相加。

那么如何求导左子树呢？，递归的调用这个求导方法就可以了。

对乘方节点的处理时比较难的。
![这里写图片描述](http://img.blog.csdn.net/20170608225954192?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGFpSmkxOTg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

先对左子树f求导，对右子树g求导。
如果f求导为0，说明是指数函数 ，如果g求导为0，说明是幂函数，分别套用公式。
至于$f(x)^g(x)$ 这种形式，求导公式有点复杂，还要去请教一些数学方面的高手。还没有做。

# 化简
求导不是最难的，最难的是化简。 比如对  1 / ( 1 + e  ^ ( - ( w * x + b ) ) )  按照上述算法求导，得到的结果是：

( 0 * ( 1 + e ^ ( - ( w * x + b ) ) ) - 1 * ( 0 + e ^ ( - ( w * x + b ) ) * 1 * ( 0 * ( w * x + b ) + - ( 1 * x + w * 0 + 0 ) ) ) ) / ( 1 + e ^ ( - ( w * x + b ) ) ) * ( 1 + e ^ ( - ( w * x + b ) ) ) 

这就需要化简。我实现了化简的几个思路：

（1） 0+x，x+0 x-0 这种化简为 x 。0*x  x*0 0/x  化简为  0
![这里写图片描述](http://img.blog.csdn.net/20170608230732429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGFpSmkxOTg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在上图中， 左图c节点为0，则应让a直接指向d。删除c和b节点。 右图为1*x的图，应让a直接指向d。
（2）x*1 1*x x/1 这种直接简化为x
（3） 两个常量进行运算，F+F， F-F， F*F， F/F 都简化为单一节点。
（4） 较为复杂的节点合并。
![这里写图片描述](http://img.blog.csdn.net/20170608231333448?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGFpSmkxOTg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在上图中，右子树有个3， 左子树有一个4，算法

如果右子树是一个常量节点，则在左子树中查找与p指向节点符号相同的节点。 经过三个星号，找到了4，然后3*4 ->12 ,随后删除原本p指向的节点，让p直接指向原本的左子树。

(5)  $x*x => x^2 $ 

(6) $ 0-x => -1*x $

(7) x^1 => x 

(8) log e - > 1

# 代码实现

本项目

# 运行测试

以 sigmoid函数为例，进行求导。

待求导的函数
1 / ( 1 + e ^ ( - ( w * x + b ) ) ) 

求导后，化简前
( 0 * ( 1 + e ^ ( - ( w * x + b ) ) ) - 1 * ( 0 + e ^ ( - ( w * x + b ) ) * 1 * ( 0 * ( w * x + b ) + - ( 1 * x + w * 0 + 0 ) ) ) ) / ( 1 + e ^ ( - ( w * x + b ) ) ) * ( 1 + e ^ ( - ( w * x + b ) ) ) 
化简后，中间还是有一个1在哪里， 问题在哪里太晚了，不查了。结果是对的。
e ^ ( - ( w * x + b ) ) * 1 * x / ( 1 + e ^ ( - ( w * x + b ) ) ) ^ 2 

# TODO

分数化简 

![这里写图片描述](http://img.blog.csdn.net/20170609101712489?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGFpSmkxOTg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)