# -*- coding: UTF-8 -*-

from 求导.ExpBak import E

x = E().sym('x')
#c = 2*x**2+3*x**4+E().float(4)**x
e = E().sym('e')
w = E().sym('w')
b = E().sym('b')
#c = 1/(1+e**(-(w*x+b)))#自定义测试表达式
# c = (w*x+b)**(w*x+b)+3*w
c = (w*e)**x
c.printme()
c.printnodes()

d = c.diff(x)
d.printme()
d.printnodes()
e = d.optm().optm()
print()
e.printme()
