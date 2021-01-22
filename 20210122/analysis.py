import pandas as pd

df = pd.read_csv('mac.csv', index_col=0)[['1']]

quantil = df.quantile([0.25, 0.5, 0.75])
mean = df.mean()
print(quantil, mean)
