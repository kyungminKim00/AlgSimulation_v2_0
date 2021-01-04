import pandas as pd
import numpy as np

file_names = ['./expectation', './y_return']
num_sample = 20000

for file_name in file_names:
    data = pd.read_csv(file_name+'.txt')
    idx = np.random.permutation(np.arange(data.shape[0]))
    tmp = data.values.squeeze()
    tmp = tmp[idx[:num_sample]]

    pd.DataFrame(tmp).to_csv(file_name+'.csv')

