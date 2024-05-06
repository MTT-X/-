# 对CAPM模型进行F检验

import pandas as pd
import numpy as np
import pickle

abspath = 'E:\金融计量学\Student Resources\Excel files\capm.xls'
data = pd.read_excel(abspath, index_col=0)

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_sandp' : LogDiff(data['SANDP']),
                    'ret_ford' : LogDiff(data['FORD']),
                    'USTB3M' : data['USTB3M']/12,
                    'ersandp' : LogDiff(data['SANDP']) - data['USTB3M']/12,
                    'erford' : LogDiff(data['FORD']) - data['USTB3M']/12})

# 代码将 data 保存到一个 pickle 文件中，然后又将其加载回来。(pickle 是一个用于序列化和反序列化Python对象的模块)。
with open(abspath + 'capm.pickle', 'wb') as handle:
    pickle.dump(data, handle)

with open(abspath + 'capm.pickle', 'rb') as handle:
    data = pickle.load(handle)

import statsmodels.formula.api as smf
# F-test: multiple hypothesis tests
formula = 'erford ~ ersandp'    # 表示对 erford 进行回归，以 ersandp 为自变量
hypotheses = 'ersandp = Intercept = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

