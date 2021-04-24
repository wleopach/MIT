import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


col_names = ["user_id", "item_id", "rating", "timestamp"]
data = pd.read_csv("data.csv", names=col_names)
data = data.drop("timestamp", 1)
data.info()
plt.hist(data["rating"])
plt.show()

train, test = train_test_split(data, test_size = 0.3)