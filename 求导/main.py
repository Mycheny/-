from sympy import *

# expr = "x**x"
expr = "log(x*w)"
expr = "(w*b)**x"
expr = "(w * x + b) ** (w * x + b) + 3 * w"
# expr = "(w*e)**(x*e)"
# expr = "x**x"
expr = "1/(1+e**(-(w*x+b)))"
expr = "1/(1+e**(-w*x))"
# expr = "1/-x"
print(expr)
x = Symbol("x")
sexpr = sympify(expr)

gs = diff(sexpr, x)
print(gs)
print(gs.subs('w', 3))
"(w*e)^x*w*e"