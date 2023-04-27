import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置字体 国标黑体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.rc('font', family='Times New Roman')

with plt.style.context(['science']):
    fig = plt.figure(figsize=(12, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    x = [29.85, 25.65, 27.99, 31.17, 28.17, 29.28, 29.41, 34.74, 31.69, 30.11, 33.22, 22.29]
    y = [77.57, 77.48, 76.75, 76.96, 77.14, 76.51, 77.36, 76.48, 77.48, 78.42, 78.75, 80.61]
    s = [103, 103, 100, 102, 101, 89, 100, 110, 200, 101, 845, 785]
    label_xy = [[0.3, 0.2], [-1.6, 0.3], [-0.7, -0.4], [0.4, -0.1], [-0.4, 0.2],
                [0.35, -0.2], [-0.8, -0.4], [-0.2, -0.35], [0.5, -0.1], [0.4, -0.1], [0.9, -0.1], [0.9, -0.1]]
    label = ['(1)Pure', u'(2)0.6$L_{Dice}$+0.4$L_{CE}$', u'(3)$L_{Focal}$', '(4)GeLU', '(5)Silu',
             '(6)Swish', '(7)Identical', '(8)Shift', '(9)Attention', '(10)LN', '(11)Aug', '(12)Mix']
    color = [np.sqrt(x[i] ** 2 + y[i] ** 2) for i in range(len(x))]
    for i in range(len(x)):
        ax.scatter(x[i], y[i], marker="o", s=6*s[i], alpha=0.4)
        ax.annotate(label[i], xy=(x[i], y[i]), xytext=(x[i]+label_xy[i][0], y[i]+label_xy[i][1]), fontsize=20)
    ax.grid(linewidth=1.2)
    ax.set_xticklabels(range(20, 37, 2), minor=False, fontsize=20)
    ax.set_yticklabels(range(76, 83, 1), minor=False, fontsize=20)
    plt.xlim([20, 36])
    plt.ylim([76, 82])
    ax.set_xlabel(r'$\bf{HD}$', fontsize=20)
    ax.set_ylabel(r'$\bf{DSC}$', fontsize=20)
    ax.spines['top'].set_linewidth(1.5)  # 设置顶部坐标轴的粗细
    ax.spines['bottom'].set_linewidth(1.5)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  # 设置右边坐标轴的粗细
    plt.show()
    # plt.savefig("不同架构设置和训练技巧的性能对比.png")
