# 对套期保值的F检验
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

abspath = 'E:\金融计量学\Student Resources\Excel files\SandPhedge.xls'
data = pd.read_excel(abspath, index_col=0)

formula = 'Spot ~ Futures'
hypotheses = 'Futures = 1'
results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_spot' : LogDiff(data['Spot']),
                    'ret_future':LogDiff(data['Futures'])})

formula = 'ret_spot ~ ret_future'
hypotheses = 'ret_future = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

