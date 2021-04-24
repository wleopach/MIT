import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from patsy import dmatrices
import statsmodels.discrete.discrete_model as sm
data = pd.read_csv("challenger-data.csv")
# subsetting data
failures = data.loc[(data.Y == 1)]
no_failures = data.loc[(data.Y == 0)]
# frequencies
failures_freq = failures.X.value_counts()#failures.groupby('X')
no_failures_freq = no_failures.X.value_counts()
# plotting
plt.scatter(failures_freq.index, failures_freq, c='red', s=40)
plt.scatter(no_failures_freq.index, np.zeros(len(no_failures_freq)), c='blue', s=40)
plt.xlabel('X: Temperature')
plt.ylabel('Number of Failures')
plt.show()
#get the data in correct format
y, X = dmatrices('Y ~ X', data, return_type='dataframe')
#build the model
logit = sm.Logit(y, X)
result = logit.fit()
# summarize the model
print(result.summary())