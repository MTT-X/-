import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

abspath = 'E:\金融计量学\Student Resources\Excel files\capm.xls'
data = pd.read_excel(abspath, index_col=0)

data.head()

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()    # 移除缺失项
    return x_diff
    
data = pd.DataFrame({'ret_sandp' : LogDiff(data['SANDP']),  # 计算对数差分收益率
                    'ret_ford' : LogDiff(data['FORD']),
                    'USTB3M' : data['USTB3M']/12,           # 将年度收益率转化为月度收益率
                    'ersandp' : LogDiff(data['SANDP']) - data['USTB3M']/12,     # 计算标普指数的超额收益率
                    'erford' : LogDiff(data['FORD']) - data['USTB3M']/12})      # 计算“福特汽车”的超额收益率

print(data.head())

plt.figure(1, dpi=150)
plt.plot(data['ersandp'], label='ersandp')
plt.plot(data['erford'], label='erford')

plt.xlabel('Date')
plt.ylabel('ersandp/erford')
plt.title('Time_series_diagram')
plt.grid(True)

plt.legend()

output_path = 'E:\金融计量学\Student Resources\Python code\chapter 5 the CAPM\graph1.png'
plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)


plt.figure(2, dpi=150)

plt.scatter(data['ersandp'], data['erford'])

plt.xlabel('ersandp')
plt.ylabel('erford')
plt.title('scatter')
plt.grid(True)

#plt.show()

output_path = 'E:\金融计量学\Student Resources\Python code\chapter 5 the CAPM\graph2.png'
plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)


formula = 'erford ~ ersandp'
results = smf.ols(formula, data).fit()
print(results.summary())


# F-test: hypothesis testing
formula = 'erford ~ ersandp'
hypotheses = 'ersandp = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

