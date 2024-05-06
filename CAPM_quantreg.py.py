import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import matplotlib.pyplot as plt

abspath = 'E:\金融计量学\Student Resources\Excel files\capm.xls'
data = pd.read_excel(abspath, index_col=0)

data = data.dropna()

def LogDiff(x):
    x_diff = 100 * np.log(x / x.shift(1))
    x_diff = x_diff.dropna()  # 移除缺失项
    return x_diff


data = pd.DataFrame({'ret_sandp': LogDiff(data['SANDP']),  # 计算对数差分收益率
                     'ret_ford': LogDiff(data['FORD']),
                     'USTB3M': data['USTB3M'] / 12,  # 将年度收益率转化为月度收益率
                     'ersandp': LogDiff(data['SANDP']) - data['USTB3M'] / 12,  # 计算标普指数的超额收益率
                     'erford': LogDiff(data['FORD']) - data['USTB3M'] / 12})  # 计算“福特汽车”的超额收益率
# regression
# quantile(50)
res = smf.quantreg('erford ~ ersandp', data).fit(q=0.5)
print(res.summary())

# Simultaneous-quantile regression
# 10 20 30 40 50 60 70 80 90, ten quantiles
quantiles = np.arange(0.10, 1.00, 0.10)

for x in quantiles:
    print('-----------------------------------------------')
    print('{0:0.01f} quantile'.format(x))
    res = smf.quantreg('erford ~ ersandp', data).fit(q=x)
    print(res.summary())

def model_paras(data, quantiles):
    '''
    这个函数的两个作用：将生成所有必要的统计量，包括常数项的 α 系数、‘ersandp’ 上的 β 系数，以及 95% 置信区间的上下界。其次，它将输出回归规格的所有拟合值
    '''
    parameters = []
    y_pred = {}
    for q in quantiles:
        res = smf.quantreg('erford ~ ersandp', data).fit(q=q)  
        # obtain regression's parameters
        alpha = res.params['Intercept']     # 常数项的a系数
        beta = res.params['ersandp']        # 变量'ersandp'的β系数
        lb_pval = res.conf_int().loc['ersandp'][0]
        ub_pval = res.conf_int().loc['ersandp'][1]
        # obtain the fitted value of y
        y_pred[q] = res.fittedvalues
        # save results to lists
        parameters.append((q,alpha,beta,lb_pval,ub_pval))
       
    quantreg_res = pd.DataFrame(parameters, columns=['q', 'alpha','beta','lb','ub'])
    y_hat = pd.DataFrame(y_pred)  
    return quantreg_res, y_hat

quantreg_paras, y_hats = model_paras(data, quantiles)

print(quantreg_paras)

y_hats.head()

plt.figure(1, dpi=150)
for i in quantreg_paras.q:
    common_index = data.index.intersection(y_hats[i].index)     # 将 x的索引和y的索引对齐
    x = data.loc[common_index, 'ersandp']
    y = y_hats[i].loc[common_index]
    if i == 0.50:
        plt.plot(x,y,color='red')        
    else:
        plt.plot(x,y,color='grey')

plt.ylabel('ersandp')
plt.xlabel('erford')
plt.show()