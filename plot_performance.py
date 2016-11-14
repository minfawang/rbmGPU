import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()

import plotly.plotly as py
from plotly.graph_objs import *

py.sign_in('voiceup', '05pez8c0ow')


Fs = [10, 20, 40, 60, 80, 100]

predict_gpu = [142.311, 277.187, 576.23, 1096.91, 1625.87, 2113.26]
train_gpu = [694.449, 1369.15, 3102.83, 6096.78, 8494.44, 11103]

predict_cpu = [306.092, 498.867, 890.282, 1121.35, 1454.49, 1786.35]
train_cpu = [2098.74, 3262.91, 5574.24, 7177.71, 9551.2, 11637.9]


y = [
    1.279,
    1.01257,
    0.970232,
    0.952474,
    0.938561,
    0.933545,
    0.9276,
    0.924411,
    0.923662,
    0.920845,
    0.920666]

plt.plot(y)
plt.xlabel("number of iterations")
plt.ylabel("RMSE")
plt.title("RBM Predicting Convergence Plot")
plt.show()
