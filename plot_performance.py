# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# sns.set(palette='Set2')
# sns.despine()

import plotly.plotly as py
from plotly.graph_objs import *

py.sign_in('voiceup', '05pez8c0ow')


Fs = [10, 20, 40, 60, 80, 100]

predict_gpu = [142.311, 277.187, 576.23, 1096.91, 1625.87, 2113.26]
train_gpu = [694.449, 1369.15, 3102.83, 6096.78, 8494.44, 11103]

predict_cpu = [306.092, 498.867, 890.282, 1121.35, 1454.49, 1786.35]
train_cpu = [2098.74, 3262.91, 5574.24, 7177.71, 9551.2, 11637.9]


trace1 = Bar(
	x = Fs,
	y = predict_cpu,
	name="CPU")
trace2 = Bar(
	x = Fs,
	y = predict_gpu,
	name="GPU"
)

data = Data([trace1, trace2])
layout = Layout(
	barmode="group",
	title="Performance Comparison on Prediction Time",
	xaxis=XAxis(
		title="number of features"
	),
	yaxis=YAxis(
		title="elapsed time (ms)"
	)
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='grouped-bar1')


trace3 = Bar(
	x = Fs,
	y = train_cpu,
	name="CPU")
trace4 = Bar(
	x = Fs,
	y = train_gpu,
	name="GPU"
)

data = Data([trace3, trace4])
layout = Layout(
	barmode="group",
	title="Performance Comparison on Training Time",
	xaxis=XAxis(
		title="number of features"
	),
	yaxis=YAxis(
		title="elapsed time (ms)"
	)
)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='grouped-bar2')






















# plt.plot(Fs, predict_cpu)
# plt.show()


# def sinplot(flip=1):
#     x = np.linspace(0, 14, 100)
#     for i in range(1, 7):
#         plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
#     plt.show()



# sinplot()