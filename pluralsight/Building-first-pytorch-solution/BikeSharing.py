# https://app.pluralsight.com/library/courses/building-your-first-pytorch-solution/table-of-contents

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
data = pd.read_csv('bike-sharing-demand/train.csv',index_col=0)



# plt.figure(figsize=(8,6))
# sns.barplot('yr','cnt',hue = 'season',data=data,ci=None)
# plt.legend(loc='upper right')
# plt.xlabel('year')
# plt.ylabel('Total bikes rented')
# plt.title('number of bikes rented per season')