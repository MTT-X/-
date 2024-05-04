import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

abspath = 'E:\金融计量学\Student Resources\Excel files\SandPhedge.xls'
data = pd.read_excel(abspath, index_col=0)

# 定义保存图片的函数
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

# 观察数据前五行
print(data.head())

formula = 'Spot ~ Futures'
results1 = smf.ols(formula, data).fit()
print(results1.summary())

# 将结果转化为字符串文本
summary_str1 = results1.summary().as_text()

# 指定图片保存路径
output_path = 'E:\金融计量学\Student Resources\Python code\chapter 3 simple linear regression\model_summary_1.png'
save_text_as_image(summary_str1,output_path)


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()    # 移除缺失项
    return x_diff
    
data = pd.DataFrame({'ret_spot' : LogDiff(data['Spot']),
                    'ret_future':LogDiff(data['Futures'])})
data.head()


data.describe()

formula = 'ret_spot ~ ret_future'
results2 = smf.ols(formula, data).fit()
print(results2.summary())

summary_str2 = results2.summary().as_text()
# 指定图片保存路径
output_path = 'E:\金融计量学\Student Resources\Python code\chapter 3 simple linear regression\model_summary_2.png'
save_text_as_image(summary_str2,output_path)


