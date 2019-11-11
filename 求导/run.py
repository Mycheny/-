from 求导.Exp import E

x = E().sym('x')
# c = 2*x**2+3*x**4+E().float(4)**x
e = E().sym('e')
w = E().sym('w')
b = E().sym('b')
# c = 1/(1+e**(-(w*x+b)))#自定义测试表达式
# c = (w * x + b) ** (w * x + b) + 3 * w
# c = w * x + b
# c = (w)**(x*e)
# c = x**x
# c = 1 / -x
c = 1/(1+e**(-w*x))

d = c.diff(x)
data = d.optm().optm()
data.generate()
print(data.result)
"e**(w*x)*w*log(e)"
