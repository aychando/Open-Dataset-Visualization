import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5.5))

sns.set(style="whitegrid", palette="muted")

dataset = pd.read_csv('/home/husein/visualization/planet/exoTrain.csv')

dataset = dataset.ix[:100, 1:15]

planet = pd.melt(dataset, "LABEL", var_name="Attributes")

swarm_plot = sns.swarmplot(x="Attributes", y="value", hue="LABEL", data = planet)

fig = swarm_plot.get_figure()
fig.savefig("planet.png")
fig.savefig("planet.pdf")