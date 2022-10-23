from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import pyDOE as pde
import torch
import numpy as np
print('hi fcs2')

def dataTransfer(W, dataType):
    a = dataType['QnShareFactors']
    b = dataType['QlShareFactors']
    c = dataType['BranchingFactors']
    d = sum(dataType['NestedFactors'])
    outputs = [W[:, :a], W[:, a:a+b], W[:, a+b:a+b+c], W[:, a+b+c:a+b+c+d]]
    return outputs

# 獲得資料 W 的 定性因子組合 定量因子組合
def level_num(W, dataType):
    c = dataTransfer(W, dataType)
    Qls = torch.cat([c[1], c[2]], dim=1)
    Qns = torch.cat([c[0], c[3]], dim=1)
    return Qls, Qns


# 繪製 3d 圖
def plot3d(X, Y, Z, xtitle=None, ytitle=None, ztitle=None, Zlim=None, cmap=cm.coolwarm):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    if Zlim:
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                            linewidth=0, antialiased=False, vmin=Zlim[0], vmax=Zlim[1])
    else:
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                            linewidth=0, antialiased=False)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    if ztitle:
        ax.set_zlabel(ztitle)
    # Customize the z axis.
    if Zlim:
        ax.set_zlim(Zlim[0], Zlim[1])

    # Add a color bar which maps values to colors.
    position = fig.add_axes([0.15, 0.2, 0.05, 0.5])#位置[左,下,右,上]
    fig.colorbar(surf, shrink=0.5, aspect=5, cax=position)
    # fig.colorbar(surf, shrink=0.5, aspect=5, location='left')

    plt.show()
    return fig, ax

# 多個 3d plot 合併再一起
def plot3ds(X, Y, Z1=None, Z2=None, Z3=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
#     X = np.concatenate([X, X, X], axis=0)
#     Y = np.concatenate([Y, Y, Y], axis=0)
#     Z = np.concatenate([Z1, Z2, Z3], axis=0)
    # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    if Z1 is not None:
        surf1 = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if Z2 is not None:
#         surf2 = ax.plot_wireframe(X, Y, Z2, rstride=10, cstride=10)
        surf2 = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if Z3 is not None:
        surf3 = ax.plot_surface(X, Y, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_zlim(-6, 8)
        
    plt.show()
    return fig

# 繪製等高線圖, trainXY 為標記初始化資料的點
def plot_contour_line(X, Y, Z, trainXY, xtitle=None, ytitle=None):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 3, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainXY is not None:
        ax.scatter(trainXY[:, 0], trainXY[:, 1])

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

# 繪製等高線圖, trainXY 為標記初始化資料的點, 多標上EI迭代後的點 及平面最小點
def plot_contour_line2(X, Y, Z, trainXY=None, EIpointsXY=None, xtitle=None, ytitle=None, minPoint=None):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 2, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainXY is not None:
        ax.scatter(trainXY[:, 0], trainXY[:, 1])

    if EIpointsXY is not None:
        ax.scatter(EIpointsXY[:, 0], EIpointsXY[:, 1], c='m')

    if minPoint is not None:
        ax.scatter(minPoint[0], minPoint[1], s=100, marker="D", c='#bcbd22', alpha=0.3)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

# 繪製等高線圖, trainX 為標記初始化資料的點, 多標上EI迭代後的點 及平面最小點, 每個點標上他是第幾個取的點
def plot_contour_line3(X, Y, Z, trainXY=None, EIpointsXY=None, xtitle=None, ytitle=None, minPoint=None, pointsNumber=None):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 2, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainXY is not None:
        ax.scatter(trainXY[:, 0], trainXY[:, 1])

    if EIpointsXY is not None:
        ax.scatter(EIpointsXY[:, 0], EIpointsXY[:, 1], c='m')

    if minPoint is not None:
        ax.scatter(minPoint[0], minPoint[1], s=100, marker="D", c='#bcbd22', alpha=0.3)

    if pointsNumber is not None:
        EN = pointsNumber
        for i in range(EN.shape[0]):
            label = str(EN[i].item()) + 'th'
            ax.annotate(label, (EIpointsXY[0][i], EIpointsXY[1][i]), fontsize=13)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

# 繪製目標函式等高線圖(以紅色繪圖)
def plot_contourf(X, Y, Z, xtitle=None, ytitle=None, Zlim=[55, 125]):
    fig, ax = plt.subplots()
    level = np.linspace(55, 130, 16)
    CS = ax.contourf(X, Y, Z, levels=level,cmap='Reds', vmin=Zlim[0], vmax=Zlim[1])
    # ax.clabel(CS, fontsize=15, inline=True)
    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    # Add a color bar which maps values to colors.
    # position = fig.add_axes([0., 0.15, 0.01, 0.5])#位置[左,下,右,上]
    colorbar = fig.colorbar(CS, location='left')
    plt.show()
    
    return fig

# 繪製折線圖 (用於 predMin 的多線折線圖繪製)
def plot_lineChart(predMin, xtitle=None, ytitle=None, between=True, color='blue'):
    fig, ax = plt.subplots()
    a, b = predMin.shape
    x = np.arange(0, b)
    for i in range(a):
        y = predMin[i, :]
        ax.plot(x, y, color=color, linewidth=0.5, alpha=0.6)

    # mean, std = predMin.mean(dim=0), predMin.std(dim=0)
    # y1 = mean + 1.96 * std
    # y2 = mean - 1.96 * std

    if between:
        y1 = np.quantile(predMin, 0.15, axis=0)
        y2 = np.quantile(predMin, 0.85, axis=0)

        c1 = list(zip(x.tolist(), x.tolist()))
        ys = list(zip(y1.tolist(), y2.tolist()))
        for i in range(x.shape[0]):
            ax.plot(c1[i], ys[i], color='red', linewidth=0.5)

    # ax.fill_between(x, y1, y2, alpha=0.1, color='green')

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)
    
    plt.show()
    return fig

# predMin 的多筆資料 line chart (包含quantile區間著色)
def plot_lineChart2(datas, lineLegend, color=['b', 'r', 'g'], xtitle=None, ytitle=None, quantile=[0.25, 0.75]):
    fig, ax = plt.subplots()
    a, b, c = datas.shape   # datas: array N * n * EIs
    x = np.arange(0, c)
    for i in range(a):
        d = datas[i, :, :]
        mean = d.mean(axis=0)
        # mean = np.quantile(d, 0.5, axis=0)
        y1 = np.quantile(d, quantile[0], axis=0)
        y2 = np.quantile(d, quantile[1], axis=0)

        ax.plot(x, mean, label=lineLegend[i], color=color[i])
        # ax.plot(x, y1, color=color[i], linestyle="--")
        # ax.plot(x, y2, color=color[i], linestyle="--")
        ax.fill_between(x, y1, y2, alpha=0.2, color=color[i])

    ax.legend()

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)
    
    plt.show()
    return fig

# predMin 的多筆資料期望值 多個結果(多個線) line chart
def plot_multiChart(datas, lineLegend, color=['b', 'r', 'g'], xtitle=None, ytitle=None, title=None):
    fig, ax = plt.subplots()
    a, b = datas.shape
    x = np.arange(0, b)
    for i in range(a):
        y = datas[i, :]
        ax.plot(x, y, label=lineLegend[i], color=color[i])
    
    ax.legend()

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    if title:
        ax.set_title(title)
    
    plt.show()
    return fig

#畫箱型圖
def box_plot(x:str, y:str, data:pd.DataFrame, labels, n):
    # print(x, y, data)
    fig = sns.catplot(kind='box', data=data, x=x, y=y, hue=x,
                dodge=False, palette=sns.color_palette("Set2"), legend_out=True, height=n, aspect=1.5)
    fig.add_legend()
    plt.setp(fig.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    fig.fig.subplots_adjust(left=0.06) # more space for the y-label
    
    return fig

# 畫 histogram 圖
def plot_histogram(datas, bins=50, xtitle=None, ytitle=None, density=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(datas, bins=bins, density=density, rwidth=0.6)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)
    
    plt.show()
    return fig

#用 LHD 生成資料點，只能生數值 factors 資料
def data_LHD(factors, samples):
    return pde.lhs(factors, samples, 'maximin')

# 初始資料點分布繪製 (配合複雜因子 Qls, Qns)
# def initiall_contour(W, Qncha, Qnfix, Qls, QnRange, dataType, ticks, xlabel, ylabel, combin=True):
#     dQls, dQns = level_num(W, dataType)
#     # return Qls, Qns
#     t1 = (((dQls == torch.tensor(Qls)) + 0.).mean(dim=1)) == 1.
#     if combin:
#         t = t1

#     else:
#         t2 = set([i for i in range(dQns.shape[1])]) - set(Qncha)
#         t2 = list(t2)
#         t2 = dQns[:, t2] == torch.tensor(Qnfix)
#         t2 = (t2 + 0.).mean(dim=1) == 1.
#         t = (t1 + 0.) + (t2 + 0.)
#         t = t == 2.

#     trainXY = dQns[t, :][:, Qncha]

#     fig, ax = plt.subplots(figsize=[8.5, 5])
#     xlim, ylim = QnRange[Qncha[0]], QnRange[Qncha[1]]
#     ax.set_xlim(*xlim)
#     ax.set_ylim(*ylim)
#     ax.set_xlabel(xlabel, fontsize=16)
#     ax.set_ylabel(ylabel, fontsize=16)
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)
#     ax.grid()

#     ax.scatter(trainXY[:, 0], trainXY[:, 1])

#     return fig

# 繪製多個圖片於一張圖中 (subplots) 沒有寫好
# def plotpics(imgs:list, picNumber:list, figsize=[15, 4.5], x1=None, x2=None):
#     a, b = picNumber[0], picNumber[1]
#     fig = plt.figure(figsize=[50, 50])
#     # ax = fig.add_subplot(a, b, 1, projection='3d')
#     # ax.plot_surface(x1, x2, imgs[0], cmap='Blues',
#     #                     linewidth=0, antialiased=False)

#     # ax = fig.add_subplot(a, b, 2, projection='3d')
#     # ax.plot_surface(x1, x2, imgs[1], cmap='Blues',
#     #                     linewidth=0, antialiased=False)  
#     for i in range(a):
#         for j in range(b):
#             index = b * i + j
#             ax = fig.add_subplot(a, b, index+1, projection='3d')
#             ax.plot_surface(x1, x2, imgs[index], cmap='Blues',
#                         linewidth=0, antialiased=False)
    
#     return fig