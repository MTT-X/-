import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def forward_selected(data, endog, exg):
    '''
    Linear model designed by forward selection based on p-values.

    Parameters:
    -----------
    data : pandas DataFrame with dependent and independent variables

    endog: string literals, dependent variable from the data
    
    exg: string literals, independent variable from the data
        
    Returns:
    --------
    res : an "optimal" fitted statsmodels linear model instance
           with an intercept selected by forward selection
    '''
    remaining = set(data.columns)
    remaining = [e for e in remaining if (e not in endog)&(e not in exg)]
    exg = [exg]
    #print(exg)

    #print(remaining)

    scores_with_candidates = []
    for candidate in remaining:
        formula = '{} ~ {}'.format(endog,' + '.join(exg + [candidate]))
        # 生成形如'endog ~ ersandp + inflation' 这样的形式，即始终以endog为因变量，然后选取不同的变量作为自变量

        score = smf.ols(formula, data).fit().pvalues[2]     # 计算不同变量进行OLS回归的P值
        scores_with_candidates.append((score, candidate))   # 将不同变量对应的P值加入字典中
    scores_with_candidates.sort()   # 将不同变量对应的P值升序排列
    #print(scores_with_candidates)

    for pval,candidate in scores_with_candidates:
        if pval < 0.2:
            exg.append(candidate)

    print(exg)

    formula = '{} ~ {}'.format(endog, ' + '.join(exg))
    res = smf.ols(formula, data).fit()
    return res

import pickle

abspath = 'E:\金融计量学\Student Resources\Excel files\macro.xls'
data = pd.read_excel(abspath, index_col=0)

# Save data to pickle
with open(abspath.replace('.xls', '.pickle'), 'wb') as handle:
    pickle.dump(data, handle)

# Load data from pickle
with open(abspath.replace('.xls', '.pickle'), 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna()

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff

data = pd.DataFrame({'dspread' : data['BMINUSA'] - data['BMINUSA'].shift(1),
                    'dcredit' : data['CCREDIT'] - data['CCREDIT'].shift(1),
                    'dprod' : data['INDPRO'] - data['INDPRO'].shift(1),
                    'rmsoft' : LogDiff(data['MICROSOFT']),
                    'rsandp' : LogDiff(data['SANDP']),
                    'dmoney' : data['M1SUPPLY'] - data['M1SUPPLY'].shift(1),
                    'inflation' : LogDiff(data['CPI']),
                    'term' : data['USTB10Y'] - data['USTB3M'],
                    'dinflation' : LogDiff(data['CPI']) - LogDiff(data['CPI']).shift(1),
                    'mustb3m' : data['USTB3M']/12,
                    'rterm' : (data['USTB10Y'] - data['USTB3M']) - (data['USTB10Y'] - data['USTB3M']).shift(1),
                    'ermsoft' : LogDiff(data['MICROSOFT']) - data['USTB3M']/12,
                    'ersandp' : LogDiff(data['SANDP']) - data['USTB3M']/12})

res = forward_selected(data,'ermsoft','ersandp')

print(res.model.formula)

print(res.summary())

print(res.rsquared_adj) # 输出调整后的R方
