import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

abspath = 'E:\金融计量学\Student Resources\Excel files\macro.xls'
data = pd.read_excel(abspath, index_col=0)
data.head()

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
print(data.head())

import pickle

with open(abspath + 'macro.pickle', 'wb') as handle:
    pickle.dump(data, handle)

formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit()
print(results.summary())

hypotheses = 'dprod = dcredit = dmoney = dspread = 0'

f_test = results.f_test(hypotheses)
print(f_test)

def save_text_as_image( summary_str, output_path):
    # 设置图形大小和分辨率
    plt.figure(figsize=(2, 1.5), dpi=100)   # 关于这里画框的选择问题，为了图片中多余的空白尽量少，尺寸大概是2 * 1.5（就目前的数据来说)
                                            # 什么是dpi: 就是“每英寸点数”，其实就是分辨率的大小，一般的显示器都是100 ~ 150，不过在打印文件的时候一般就是300~600甚至更高

    # 选择一种等宽字体
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 10}

    # 应用字体
    plt.rc('font', **font)
    # "**" 在函数调用时用于将字典的内容作为关键字参数传递给函数。这种语法称为“关键字参数展开”，允许你将一个包含键值对的字典直接转换为函数的命名参数。

    # 将文本添加到图形
    plt.text(0.01, 0.5, summary_str, fontsize=10, va='top', ha='center')

    # 移除坐标轴
    plt.axis('off')

    # 保存图形到本地
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    # bbox_inches = inches: 尝试找出紧边界框

    # 关闭图形以释放内存
    plt.close()

summary_str = results.summary().as_text()
output_path = 'E:\金融计量学\Student Resources\Python code\chapter 7 the APT and step-wise regression\ols_model_summary.png'
save_text_as_image(summary_str,output_path)