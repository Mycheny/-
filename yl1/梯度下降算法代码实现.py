import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# Read data from csv
pga = pd.read_csv("pga.csv")
# Normalize the data 归一化值 (x - mean) / (std)

pga.distance = np.linspace(300, 0, 197) + np.random.random(197)*50
pga.accuracy = np.linspace(100, 200, 197) + np.random.random(197)*50

pga.distance = (pga.distance - pga.distance.mean()) / pga.distance.std()
pga.accuracy = (pga.accuracy - pga.accuracy.mean()) / pga.accuracy.std()
print(pga.head())
plt.scatter(pga.distance, pga.accuracy)
plt.xlabel('normalized distance')
plt.ylabel('normalized accuracy')
plt.show()
# ### accuracy = $\theta_1$ $distance_i$ + $\theta_0$ + $\alpha$
# - $\theta_0$是bias
# In[4]:
# accuracyi=θ1distancei+θ0+ϵ
from sklearn.linear_model import LinearRegression
import numpy as np

# We can add a dimension to an array by using np.newaxis
print("Shape of the series:", pga.distance.shape)
print("Shape with newaxis:", pga.distance[:, np.newaxis].shape)
# The X variable in LinearRegression.fit() must have 2 dimensions
lm = LinearRegression()
lm.fit(pga.distance[:, np.newaxis], pga.accuracy)
theta1 = lm.coef_[0]
print(theta1)


# ### accuracy = $\theta_1$ $distance_i$ + $\theta_0$ + $\alpha$
# - $\theta_0$是bias
# - #### 没有用梯度下降来求代价函数
# In[9]:
# The cost function of a single variable linear model
# 单变量 代价函数
def cost(theta0, theta1, x, y):
    # Initialize cost
    J = 0
    # The number of observations
    m = len(x)
    # Loop through each observation
    # 通过每次观察进行循环
    for i in range(m):
        # Compute the hypothesis
        # 计算假设
        h = theta1 * x[i] + theta0
        # Add to cost
        J += (h - y[i]) ** 2
    # Average and normalize cost
    J /= (2 * m)
    return J


# The cost for theta0=0 and theta1=1
print(cost(0, 1, pga.distance, pga.accuracy))
theta0 = 100
theta1s = np.linspace(-3, 2, 100)
costs = []
for theta1 in theta1s:
    costs.append(cost(theta0, theta1, pga.distance, pga.accuracy))
plt.plot(theta1s, costs)
plt.show()
# In[6]:
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# # Example of a Surface Plot using Matplotlib
# # Create x an y variables
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# # We must create variables to represent each possible pair of points in x and y
# # ie. (-10, 10), (-10, -9.8), ... (0, 0), ... ,(10, 9.8), (10,9.8)
# # x and y need to be transformed to 100x100 matrices to represent these coordinates
# # np.meshgrid will build a coordinate matrices of x and y
# X, Y = np.meshgrid(x, y)
# # print(X[:5,:5],"\n",Y[:5,:5])
# # Compute a 3D parabola
# Z = X ** 2 + Y ** 2
# # Open a figure to place the plot on
# fig = plt.figure()
# # Initialize 3D plot
# ax = Axes3D(fig)
# # Plot the surface
# ax.plot_surface(X=X, Y=Y, Z=Z)
# plt.show()
# Use these for your excerise
theta0s = np.linspace(-2, 2, 30)
theta1s = np.linspace(-2, 2, 30)
COST = np.empty(shape=(30, 30))
# Meshgrid for paramaters
T0S, T1S = np.meshgrid(theta0s, theta1s)
# for each parameter combination compute the cost
for i in range(30):
    if i % 10 == 0:
        print(i)
    for j in range(30):
        COST[i, j] = cost(T0S[0, i], T1S[j, 0], pga.distance, pga.accuracy)
# make 3d plot
print("aaaa")
fig2 = plt.figure()
ax1 = Axes3D(fig2)

plt.xlabel('theta0s',fontsize=12,color='red',rotation=60,verticalalignment='top')
plt.ylabel('theta1s',fontsize=14,color='blue',rotation=30,horizontalalignment='center')
ax1.plot_surface(X=T0S, Y=T1S, Z=COST)

point = (2, 2)
x = np.linspace(point[0]-0.13333, point[0]+0.13333, 2)
y = np.linspace(point[1]-0.13333, point[1]+0.13333, 2)
x, y = np.meshgrid(x, y)
z = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        z[i, j] = cost(x[0, i], y[j, 0], pga.distance, pga.accuracy)
ax1.plot_surface(X=x, Y=y, Z=z)
plt.show()


# ### 求导
# In[21]:
# 对 theta1 进行求导
def partial_cost_theta1(theta0, theta1, x, y):
    # Hypothesis
    h = theta0 + theta1 * x
    # Hypothesis minus observed times x
    diff = (h - y) * x
    # Average to compute partial derivative
    partial = diff.sum() / (x.shape[0])
    return partial


partial1 = partial_cost_theta1(0, 5, pga.distance, pga.accuracy)
print("partial1 =", partial1)


# 对theta0 进行求导
# Partial derivative of cost in terms of theta0
def partial_cost_theta0(theta0, theta1, x, y):
    # Hypothesis
    h = theta0 + theta1 * x
    # Difference between hypothesis and observation
    diff = (h - y)
    # Compute partial derivative
    partial = diff.sum() / (x.shape[0])
    return partial


partial0 = partial_cost_theta0(1, 1, pga.distance, pga.accuracy)
print("partial0 =", partial0)


# ### 梯度下降进行更新
# In[22]:
# x is our feature vector -- distance
# y is our target variable -- accuracy
# alpha is the learning rate
# theta0 is the intial theta0
# theta1 is the intial theta1
def gradient_descent(x, y, alpha=0.1, theta0=-1, theta1=1):
    max_epochs = 1000  # Maximum number of iterations 最大迭代次数
    counter = 0  # Intialize a counter 当前第几次
    c = cost(theta1, theta0, pga.distance, pga.accuracy)  ## Initial cost 当前代价函数
    costs = [c]  # Lets store each update 每次损失值都记录下来
    # Set a convergence threshold to find where the cost function in minimized
    # When the difference between the previous cost and current cost
    #        is less than this value we will say the parameters converged
    # 设置一个收敛的阈值 (两次迭代目标函数值相差没有相差多少,就可以停止了)
    convergence_thres = 0.0000000000000001
    cprev = c + 10
    theta0s = [theta0]
    theta1s = [theta1]
    # When the costs converge or we hit a large number of iterations will we stop updating
    # 两次间隔迭代目标函数值相差没有相差多少(说明可以停止了)
    while (np.abs(cprev - c) > convergence_thres) and (counter < max_epochs):
        cprev = c
        # Alpha times the partial deriviative is our updated
        # 先求导, 导数相当于步长
        update0 = alpha * partial_cost_theta0(theta0, theta1, x, y)
        update1 = alpha * partial_cost_theta1(theta0, theta1, x, y)
        # Update theta0 and theta1 at the same time
        # We want to compute the slopes at the same set of hypothesised parameters
        #             so we update after finding the partial derivatives
        # -= 梯度下降，+=梯度上升
        theta0 -= update0
        theta1 -= update1

        # Store thetas
        theta0s.append(theta0)
        theta1s.append(theta1)

        # Compute the new cost
        # 当前迭代之后，参数发生更新
        c = cost(theta0, theta1, pga.distance, pga.accuracy)
        # Store updates，可以进行保存当前代价值
        costs.append(c)
        counter += 1  # Count
    print(counter)

    # 将当前的theta0, theta1, costs值都返回去
    return {'theta0': theta0, 'theta1': theta1, "costs": costs}


res = gradient_descent(pga.distance, pga.accuracy)

print("Theta0 =", ['theta0'])
print("Theta1 =", res['theta1'])
print("costs =", res['costs'])
descend = gradient_descent(pga.distance, pga.accuracy, alpha=.01)
plt.scatter(range(len(descend["costs"])), descend["costs"])
plt.show()
