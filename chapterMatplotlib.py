# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
x = np.linspace(0, 10, 100)

# %%
fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

# %%
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

# %%
fig = plt.figure()
ax = plt.axes()

# %%
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), color="blue")
ax.plot(x, np.cos(x), color="#0099ff")
fig

# %%
plt.plot(x, np.sin(x - 0), color='blue')        
plt.plot(x, np.sin(x - 1), color='g')           
plt.plot(x, np.sin(x - 2), color='0.75')        
plt.plot(x, np.sin(x - 3), color='#FFDD44')     
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))
plt.plot(x, np.sin(x - 5), color='chartreuse') 

# %%
plt.plot(x, x+4, linestyle="--")
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':'); # dotted

# %%
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

# %%
plt.plot(x, np.sin(x))
plt.xlim(10, -1)
plt.ylim(-1.2, 1.2)

# %%
plt.plot(x, np.sin(x))
plt.axis('tight');

# %%
plt.plot(x, np.sin(x))
plt.title("A Sine curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

# %%
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, "o", color="red")

# %%
plt.plot(x, y, "-o", color="black")

# %%
plt.scatter(x, y, marker="o")

# %%
rng = np.random.default_rng(0)
x = rng.normal(size=100)
y = rng.normal(size=100)
colors = rng.random(100)
sizes = 1000 * rng.random(100)

plt.scatter(x, y, c=colors, sizes=sizes, alpha=0.3)
plt.colorbar()

# %%
from sklearn.datasets import load_iris

iris = load_iris()

features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.4, s=100*features[3], c=iris.target, cmap="viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);

# %%
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt=".k")

# %%
plt.errorbar(
    x, y, yerr=dy, fmt="o", color="black", ecolor="lightgray", elinewidth=3, capsize=0
)

# %%
from sklearn.gaussian_process import GaussianProcessRegressor

# define model and draw some data

model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# compute the gaussian process fit
gp = GaussianProcessRegressor()
gp.fit(xdata[:, np.newaxis], ydata)


xfit = np.linspace(0, 10, 1000)
yfit, dyfit = gp.predict(xfit[:, np.newaxis], return_std=True)

# %%
plt.plot(xdata, ydata, "or")
plt.plot(xfit, yfit, "-", color="gray")
plt.fill_between(xfit, yfit -dyfit, yfit+dyfit, color="gray", alpha=0.3)
plt.xlim(0, 10)

# %%
def f(x, y):
  return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# %%
x = np.linspace(0, 5, 40)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)

Z = f(X, Y)
plt.contour(X, Y, Z, colors="black")

# %%
plt.contour(X, Y, Z, 20, cmap="RdGy")

# %%



