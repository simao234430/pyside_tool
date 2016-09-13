# -*- coding: utf-8 -*-
# from pyExcelerator import *
# from mayavi import mlab
import platform
import sys
import zipfile
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from xlsxwriter import *
from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
from matplotlib import font_manager
import traceback
from scipy import stats

import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import os
from numpy.random import randn
from django.conf import settings
from pylab import *
import matplotlib.patches as mpatches

if platform.system() == "Linux":
    zhfont1 = matplotlib.font_manager.FontProperties(fname=u'/usr/share/fonts/华文细黑.ttf')
elif platform.system() == "Darwin":
    zhfont1 = matplotlib.font_manager.FontProperties(fname=u'/Library/Fonts/华文细黑.ttf')


def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def covert_image_format(infile):
    f, e = os.path.splitext(infile)
    outfile = f + ".bmp"
    if infile != outfile:
        try:
            Image.open(infile).convert("RGB").save(outfile)
        except IOError:
            print("cannot convert", infile)

            # fig = plt.figure()
            # file_in = file
            # file_out = file_in.replace('.png','.bmp')
            # fig.savefig(file_in)
            # img = Image.open(file_in)
            # img.load() # 这句不可少，否则会出错
            # if len(img.split()) == 4:
            #    r, g, b, a = img.split()
            #    img = Image.merge('RGB', (r, g, b))
            # img.save(file_out)


# covert_image_format("test.png")
def data1to6(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    x = zeros((numberOfLines, 1))
    print x
    y = zeros((numberOfLines, 1))
    z = zeros((numberOfLines, 1))
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        # print index
        # print listFromLine
        x[index, :] = listFromLine[2]
        y[index, :] = listFromLine[3]
        z[index, :] = listFromLine[4]
        index += 1
    xMean = mean(x)
    yMean = mean(y)
    zMean = mean(z)
    px, t1 = histogram(x, bins=100, normed=True)
    t1 = (t1[:-1] + t1[1:]) / 2
    py, t2 = histogram(y, bins=100, normed=True)
    t2 = (t2[:-1] + t2[1:]) / 2
    pz, t3 = histogram(z, bins=100, normed=True)
    t3 = (t3[:-1] + t3[1:]) / 2
    fig = plt.figure()
    pl.plot(t1, px, label="x")
    pl.plot(t2, py, 'r', label="y")
    pl.plot(t3, pz, 'g', label="z")
    plt.xlabel('cm');
    plt.ylabel('pdf');
    plt.legend()
    plt.show()


use_cols = [2, 3, 4, 5]


# return 4路数据
def parse_postion_data(file_path):
    # use_cols = [0,1,2,3]
    # header_names = ['X','Y','Z','D'],
    # f = pd.read_csv(file_path,usecols = use_cols,header=None,sep='\s+')
    f = pd.read_csv(file_path, usecols=use_cols, header=None)
    return f
    # print f
    # print f.mean()
    # ndarray = f[0]
    # ndarray = f[0]
    # print ndarray
    # print size(ndarray)
    # #mean 均值
    # #print type(f[2])
    # print f[2].min()
    # print f[2].max()
    # print ndarray.mean()
    # print ndarray.std()
    # print ndarray.var()
    # print f.ix[0:,1:2]
    # print f.ix[0:1:,0:]
    # print type(f)


def add_image_out(ws, data, dir_path, min_, max_):
    # generate_3Dimage(data)
    # return

    generate_xyzd_image(data, dir_path, min_, max_)
    # generate_image(data,'d')
    # covert_image_format("percent.png")
    # ws.insert_bitmap('percent.bmp', 20, 6)
    # ws.insert_image(20, 6,'percent.png')

    # generate_image_1plus6(data)
    # covert_image_format("1plus6.png")
    # ws.insert_image(70, 6,'1plus6.png')
    # ws.insert_bitmap('1plus6.bmp', 70, 6)


def generate_image_1plus6(data):
    # print type(data)
    # print data.columns

    x = data[2]
    y = data[3]
    z = data[4]
    d = data[5]
    xMean = mean(x)
    yMean = mean(y)
    zMean = mean(z)
    dMean = mean(d)
    px, t1 = histogram(x, bins=100, normed=True)
    t1 = (t1[:-1] + t1[1:]) / 2
    py, t2 = histogram(y, bins=100, normed=True)
    t2 = (t2[:-1] + t2[1:]) / 2
    pz, t3 = histogram(z, bins=100, normed=True)
    t3 = (t3[:-1] + t3[1:]) / 2
    pd, t4 = histogram(d, bins=100, normed=True)
    t4 = (t4[:-1] + t4[1:]) / 2
    fig = plt.figure()
    pl.plot(t1, px, label="x")
    pl.plot(t2, py, 'r', label="y")
    pl.plot(t3, pz, 'g', label="z")
    pl.plot(t4, pz, 'b', label="d")
    plt.xlabel('cm')
    # plt.ylabel('pdf')
    # plt.legend()
    plt.savefig('1plus6.png', format='png')
    # plt.show()


def get_core_3d_rate(data, distance):
    # print df
    # print "data:",type(data)
    count = 0
    origin_x = data[start_index + 0].mean()
    origin_y = data[start_index + 1].mean()
    origin_z = data[start_index + 2].mean()
    origin = np.array((origin_x, origin_y, origin_z))
    print origin
    for row_index, row in data.iterrows():
        point = np.array((row[2], row[3], row[4]))
        if np.linalg.norm(point - origin) < distance:
            count = count + 1
    print count
    print float(count) / float(data.shape[0])
    return float(count) / float(data.shape[0]) * 100


def generate_3Dimage(data, dir_path, image_type=None):
    rate = get_core_3d_rate(data, 2)
    fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(111, projection='3d')

    #    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #        #xs = randrange(n, 23, 32)
    #        #ys = randrange(n, 0, 100)
    #        #zs = randrange(n, zl, zh)
    #        xs = data[2]
    #        ys = data[3]
    #        zs = data[4]
    #        ax.scatter(xs, ys, zs, c=c, marker=m)

    # xs = data[2]
    # ys = data[3]
    # zs = data[4]

    # surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)


    x = data[start_index + 0]
    y = data[start_index + 1]
    z = data[start_index + 2]

    ax = fig.gca(projection='3d')
    # ax.set_xlim3d(x.min(), x.max())
    # ax.set_ylim3d(y.min(), y.max())
    # ax.set_zlim3d(z.min(), z.max())

    limit = 3.5
    ax.set_xlim3d(x.mean() - limit, x.mean() + limit)
    ax.set_ylim3d(y.mean() - limit, y.mean() + limit)
    ax.set_zlim3d(z.mean() - limit, z.mean() + limit)

    xyz = np.vstack([x, y, z])
    density = stats.gaussian_kde(xyz)(xyz)

    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    ax.scatter(x, y, z, c=density, s=6)
    # plt.show()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    diff = 2
    x3 = x.mean() + diff * np.outer(np.cos(u), np.sin(v))
    y3 = y.mean() + diff * np.outer(np.sin(u), np.sin(v))
    z3 = z.mean() + diff * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x3, y3, z3, rstride=4, cstride=4, color='r', alpha=0.7)

    green_patch = mpatches.Patch(color='green', label=u'误差小于2cm所占百分比:%s' % rate)
    plt.legend(handles=[green_patch])
    # plt.show()
    # plt.savefig('3D.png', format='png')
    if image_type == "png":
        plt.savefig(dir_path + '3D.png', dpi=1000, format='png')
    else:
        plt.savefig(dir_path + '3D.png', dpi=1000, format='png')
        plt.savefig(dir_path + '3D.svg', format='svg', dpi=1200)
    plt.close()


start_index = 2
description_map_str = {
    # order :title ,x ,y
    "x": (u'x坐标位置分析', u"x坐标位置", u"该位置所占比重", start_index + 0),
    "y": (u'y坐标位置分析', u"y坐标位置", u"该位置所占比重", start_index + 1),
    "z": (u'z坐标位置分析', u"z坐标位置", u"该位置所占比重", start_index + 2),
    "d": (u'探测距离波动分布图', u"距离探测波动大小", u"波动大小值所占比重", start_index + 3)
}


def generate_xyzd_image(data, dir_path, min_, max_):
    image = None
    for i in description_map_str:
        generate_image(data, i, dir_path, min_, max_, image)
        # generate_image(data,i)
    generate_3Dimage(data, dir_path, image)
    # generate_3Dimage(data)


def generate_percent_image(percent_map, image_type=None):
    print percent_map
    plt.plot(percent_map.keys(), percent_map.values())
    plt.hist(percent_map.keys())
    # plt.show()
    if image_type == "png":
        plt.savefig('percent.png', dpi=1000, format='png')
    else:
        plt.savefig('percent.png', dpi=1000, format='png')
        plt.savefig('percent.svg', format='svg', dpi=1200)
    plt.close()


def generate_image(data, key, dir_path, min_, max_, image_type=None):
    # style set 这里只是一些简单的style设置
    # sns.set_palette('deep', desat=.6)
    # sns.set_context(rc={'figure.figsize': (8, 5) } )

    # plt.figure(figsize=(9,5))
    # result_map, percent_map = get_distance_offset_statistics(data[5])
    # plt.plot(percent_map.keys(), percent_map.values())
    ##plt.hist(percent_map.keys())
    ##plt.show()
    description_list = description_map_str[key]
    index = description_list[3]

    plt.figure(figsize=(11, 6))
    length = float(data.size)
    mean = data[index].mean()
    std1 = len(filter(lambda x: x > mean - 1 and x < mean + 1, data[index])) / length * 100
    std2 = len(filter(lambda x: x > mean - 2 and x < mean + 2, data[index])) / length * 100
    red_patch = mpatches.Patch(color='red', label=u'avg:%s' % data[index].mean())
    blue_patch = mpatches.Patch(color='blue', label=u'std:%s' % data[index].std())
    green_patch = mpatches.Patch(color='green', label=u'var:%s' % data[index].var())
    mix_patch = mpatches.Patch(color='yellow', label=u'1std:%s' % (std1))
    mix2_patch = mpatches.Patch(color='yellow', label=u'2std:%s' % (std2))
    # red_patch = mpatches.Patch(color='red', label=u'均值:方差:')
    plt.title(description_list[0], fontproperties=zhfont1)
    plt.xlabel(description_list[1], fontproperties=zhfont1)
    plt.ylabel(description_list[2], fontproperties=zhfont1)
    # plt.xlim(0,data[index].max() + 0.1)
    plt.xlim(max(data[index].min() - 0.1, 0), data[index].max() + 0.1)
    # plt.savefig('percent.png', format='png')
    # plt.legend([red_dot,  white_cross], ["均值:", "方差:"])

    # if data[index].min() < 0:
    #    raise Exception('error distance data')
    sns.kdeplot(data[index], color='#FF0000', shade=True, lw=1, legend=True)
    plt.legend(handles=[red_patch, blue_patch, green_patch, mix_patch, mix2_patch])

    l = plt.axvline(data[index].mean(), color='red')
    l = plt.axvline(data[index].mean() + 1)
    l = plt.axvline(data[index].mean() - 1)
    l = plt.axvline(data[index].mean() + 2)
    l = plt.axvline(data[index].mean() - 2)

    # 1 cm 精度
    mu, sigma = data[index].mean(), 1
    # mu, sigma = data[index].mean(),data[index].std()
    for i, c in zip([min_, max_], ['#00FF00', "#C1F320"]):
        s = np.random.normal(mu, i, 90000)
        sns.kdeplot(s, color=c, shade=True, lw=1, legend=True)
    # plt.show()
    if image_type == "png":
        plt.savefig(dir_path + '%s.png' % key, dpi=1000, format='png')
    else:
        plt.savefig(dir_path + '%s.png' % key, dpi=1000, format='png')
        plt.savefig(dir_path + '%s.svg' % key, format='svg', dpi=1200)
    plt.close()
    # plt.close()


# return 均值 方差
def get_mean_and_var(narray):
    return narray.mean(), narray.var(), narray.std()


def get_distance_offset_statistics(narray):
    min = 0
    max = int(narray.max()) + 1
    # print narray.size
    result_map = {}
    percent_map = {}
    i = 0
    while i < max + 1:
        result_map[i] = 0
        i = i + 1
    for j in narray:
        # print i
        temp = int(j)
        result_map[temp] = int(result_map[temp]) + 1
    i = 0
    while i < max + 1:
        percent_map[i] = (float)(result_map[i]) / narray.size * 100
        i = i + 1
    return result_map, percent_map



    # factor = pd.cut(narray, range(max+1))
    # print factor
    # temp = pd.value_counts(factor)
    ##frame.assign(ratio = frame[1]/temp.values.sum())
    # print type(temp)
    # print temp.keys
    # print temp.values
    # print stats.scoreatpercentile(temp.values,range(max+1))


def get_mean_and_var_from_dataframe(data):
    result = {}
    result_mean = []
    result_var = []
    for i in use_cols:
        narray = data[i]
        # print narray
        result_mean.append(narray.mean())
        result_var.append(narray.var())
    result['mean'] = result_mean
    result['var'] = result_var
    # print result
    return result


def add_percent_out(ws, data):
    result_map, percent_map = get_distance_offset_statistics(data[5])

    # generate_percent_image(percent_map,None)

    x_start = 7
    y_start = 4
    ws.write(x_start, y_start, unicode("精度", 'utf-8'))
    ws.write(x_start + 1, y_start, unicode("次数", 'utf-8'))
    ws.write(x_start + 2, y_start, unicode("百分比", 'utf-8'))
    size = len(result_map)

    # for i in range(size):
    # print result_map.keys()
    for i in result_map.keys():
        ws.write(x_start, y_start + i + 1, i)
        ws.write(x_start + 1, y_start + i + 1, result_map[i])
        ws.write(x_start + 2, y_start + i + 1, percent_map[i])


def add_mean_var_out(ws, data):
    mean_var_result = get_mean_and_var_from_dataframe(data)
    x_start = 0
    y_start = 4
    ws.write(x_start, y_start + 1, unicode("均值", 'utf-8'))
    ws.write(x_start, y_start + 2, unicode("方差", 'utf-8'))
    ws.write(x_start + 1, y_start, unicode("X", 'utf-8'))
    ws.write(x_start + 2, y_start, unicode("Y", 'utf-8'))
    ws.write(x_start + 3, y_start, unicode("Z", 'utf-8'))
    ws.write(x_start + 4, y_start, unicode("D", 'utf-8'))

    for i in range(4):
        ws.write(x_start + i + 1, y_start + 1, mean_var_result['mean'][i])
        ws.write(x_start + i + 1, y_start + 2, mean_var_result['var'][i])


def add_raw_input(ws, data):
    first_index = use_cols[0]
    data_size = size(data[first_index])
    ws.write(0, 0, "x")
    ws.write(0, 1, "y")
    ws.write(0, 2, "z")
    ws.write(0, 3, "D")
    for i in range(0, data_size):
        ws.write(i + 1, 0, data[first_index + 0][i])
        ws.write(i + 1, 1, data[first_index + 1][i])
        ws.write(i + 1, 2, data[first_index + 2][i])
        ws.write(i + 1, 3, data[first_index + 3][i])


def add_out_data(ws, data):
    add_raw_input(ws, data)
    add_mean_var_out(ws, data)
    add_percent_out(ws, data)


def out_analysis_result(data, dir_path, min_, max_):
    try:
        w = Workbook(dir_path + 'analysis_positon_result.xls')
        ws = w.add_worksheet('')
        add_out_data(ws, data)
        add_image_out(ws, data, dir_path, min_, max_)
        # f = w.save(dir_path+'analysis_positon_result.xls')
        w.close()
    except Exception, e:
        print "excp", e
        traceback.print_exc()


def generate_zipfile(dir_path):
    f = zipfile.ZipFile(dir_path + 'png.zip', 'w', zipfile.ZIP_DEFLATED)
    print "zifile", dir_path
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            print filename
            # if os.path.splitext(str(filename))[1] == 'png':
            if filename.endswith('png'):
                # f.write(os.path.join(dirpath,filename))
                f.write(dir_path + filename)
    f.close()


# generate_zipfile('/home/xiaojun/python_code/tool/data_analysis/kunchen/static/ca8b1ad0-530b-11e6-b8b9-00163e002270/')
def analysis_postion_data(file_path, dir_path, min_precision, max_precision):
    # data = parse_postion_data(file_path)
    data = parse_postion_data(dir_path + 'aa.csv')
    # get_distance_offset_statistics(data[5])
    analysis_result = get_mean_and_var_from_dataframe(data)
    out_analysis_result(data, dir_path, min_precision, max_precision)
    generate_zipfile(dir_path)

    # print analysis_result

# analysis_postion_data("./aa.csv",'/Users/simao/python_code/kunchen/kunchen' +'/')analysis_postion_data("./aa.csv",'/Users/simao/python_code/kunchen/kunchen' +'/static/')
# analysis_postion_data("./aa.csv",'/Users/simao/python_code/kunchen/kunchen' +'/static/')
# data1to6("../aa.csv")
# analysis_postion_data("./aa.csv",'/Users/simao/python_code/kunchen/kunchen' +'/static/')
