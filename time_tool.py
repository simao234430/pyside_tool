# -*- coding: utf-8 -*-
# from pyExcelerator import *
import re
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
import matplotlib.patches as mpatches

import pylab as pl
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import os
from numpy.random import randn
from itertools import *

import sys

from tool import covert_image_format, get_mean_and_var, zhfont1, generate_zipfile
# import tool
import time
from functools import wraps


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print "@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__)
        back = func(*args, **args2)
        print "@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__)
        print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back

    return newFunc


def log_time_delta(func):
    @wraps(func)
    def deco():
        start = datetime.now()
        res = func(*args)
        end = datetime.now()
        delta = end - start
        print("func runed ", delta)
        return res

    return deco


def block(file, size=65536):
    while True:
        nb = file.read(size)
        if not nb:
            break
        yield nb


def getLineCount(filename):
    with open(filename, "r", buffering=-1) as f:
        return sum(line.count("\n") for line in block(f))


class FileNeedUploadException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "分析文件没有输入,请检查"


class FileException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "文件输入格式不对，请检查"


class LineCountException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "基站总个数和文件输入不符合，请检查"


# class Timer(object):
#    def __init__(self, verbose=False):
#        self.verbose = verbose
#
#    def __enter__(self):
#        self.start = time.time()
#        return self
#
#    def __exit__(self, *args):
#        self.end = time.time()
#        self.secs = self.end - self.start
#        self.msecs = self.secs * 1000  # millisecs
#        if self.verbose:
#            print 'elapsed time: %f ms' % self.msecs
#
# def time_profile(func):
#    def callf(*args, **kwargs):
#        with Timer() as t:
#            res = func(*args, **kwargs)
#        print getattr(func,'__name__'), t.secs
#        return res
#    return callf

def generate_compare(index_li):
    result = []
    for i in combinations(index_li, 2):
        result.append(i)
    #print result
    return result
unit = 33.333
count_size = 100  # 分析时间戳差分布所取步长
# number = 6       #基站数量
# index = 7        #时间戳差最开始的列数字

# number = 4       #基站数量
# index = 23        #时间戳差最开始的列数字
# 下面三行生成所用到的数据列的选择下标
max_value = 40


# col = range(max_value)
# use_cols = col[index:index + number]
# print use_cols


# 生成一个list 包含 组合 元组 类似这样的结构 [(0, 1), (0, 2) ....]
def generate_list_item(num=6):
    result = []
    for i in combinations(range(num), 2):
        result.append(i)
    # print result
    return result


# 按照key值排序
def sortedDictValues(adict):
    keys = adict.keys()
    keys.sort()
    return [adict[key] for key in keys]


def two_list_sub(l1, l2):
    return list(map(lambda (a, b): a - b, zip(l2, l1)))


def parse_timestamp_data(file, dir, start, count):
    # 读入csv 源文件  use_cols 选取要操作的列 过滤其他不需要的列
    try:
        print start, count
        start_para = int(start)
        end_para = int(start) + int(count)
        print start_para, end_para
        col = range(max_value)
        use_cols = col[start_para:end_para]
        print "use_cols", use_cols
        df = pd.read_csv(file, usecols=use_cols, header=None)
        print df
        # 下面2行 由于保留小数点后面三位 整体转换为int类型 方便计算
        float_data = df.applymap(lambda x: x * 1000)
        data = float_data.astype('int64')
    except Exception, e:
        print "excp", e
        traceback.print_exc()
        raise FileException
    # global_min =int(data.min().min())
    # sort_data = data.sort_index(axis = 1)
    # print sort_data
    # print data[index].size
    #
    count_map = {}

    count_map_detail = {}
    count_map_result = {}
    # start_para 只是为了行遍历
    for i in range(data[start_para].size):
        temp_map = {}
        for j in use_cols:
            # temp_map 前面是时间差 后面是基站序号
            # {239421893: 11, 239420911: 8, 239417008: 12, 239420273: 9, 239422645: 10, 239421721: 7}
            temp_map[data[j][i]] = j
            # print data[j][i]
        # dict= sorted(map.items(), key=lambda d:d[0], reverse = False)
        # index_map 是时间差排序序号的一个列表 如果存在 计数+1 否则创建1

        index_map = sortedDictValues(temp_map)
        if str(index_map) in count_map.keys():
            count_map[str(index_map)] = count_map[str(index_map)] + 1
        else:
            count_map[str(index_map)] = 1
            # 时间差排序列表 对应那些行在这个序列里面
            count_map_detail[str(index_map)] = []
            count_map_result[str(index_map)] = index_map
        count_map_detail[str(index_map)].append(i)

        # key = lambda d:d[0]
        # for i in map.iteritems():
        #    print key(i),   #输出31 3 0 5 56 4，这些都是字典dic的值
    # print count_map_detail
    # print count_map.keys()
    # 选取出现次数最多的 时间差排序 字典
    data_map_index = sorted(count_map.items(), key=lambda d: d[1], reverse=True)[0][0]
    # data_map 就是次数出现最多的 {239421893: 11, 239420911: 8, 239417008: 12, 239420273: 9, 239422645: 10, 239421721: 7}

    # data_map 是排序后的原来逻辑序号
    data_map = count_map_result[data_map_index]
    print data_map

    # data_sheet_list 是有序的 通过data_map 去找 已经排好序的 下标 也可以保障有序
    data_sheet_list = generate_list_item(int(count))
    print "data_sheet_list", data_sheet_list
    data_sheet_input = {}
    # global_min = sys.maxint
    # global_max= int(data.max().max()) - int(data.min().min()) + 1
    # global_max = 0
    for i in data_sheet_list:
        # data_sheet_input[str(i)] = data[data_map[
        # print data[data_map[
        small = i[0]
        big = i[1]
        print small, big
        s = "table" + str(small) + "_" + str(big)
        print s
        x = []
        v1 = data[data_map[small]].tolist()
        v2 = data[data_map[big]].tolist()
        data_sheet_input[s] = two_list_sub(v1, v2)
        # min_temp1 = two_list_sub(v1,v2)
        # min_temp = min(filter(lambda x:x>0,min_temp1))
        # if global_min > min_temp:
        #    global_min = min_temp
        # if global_max < max(min_temp1):
        #    global_max = max(min_temp1)
        # if global_min < 0:
        #    raise NameError("min 为负数 出错")
        # temp = list(map(lambda(a,b):a-b, zip(v2,v1)))
        # temp = map(lambda x: y-z, zip(v2, v1))
    # data_sheet_input 这时候就是时间戳差  count_map_detail[data_map_index]
    # 就是误差数据 在下面的处理中要过滤掉的
    # print data_sheet_input
    return data_sheet_input, count_map_detail[data_map_index], data_map


def get_max(data):
    max_data = 0
    for i in data:
        if i > max_data:
            max_data = i
    return max_data


def get_min(data):
    min_data = sys.maxint
    for i in data:
        if i < min_data:
            min_data = i
    return min_data


def get_timestamp_diff_statistics(col_data, global_min, global_max):
    # print "col_data type",type(col_data)
    max_value, min_value = global_max + 1, global_min - 1
    # max_value, min_value = get_max(col_data) + 1 ,get_min(col_data) - 1
    count_map = {}
    percent_map = {}
    show_map = {}

    step = float((float(max_value) - float(min_value)) / count_size)
    # print "*****",max_value ,min_value,count_size,step


    # print "here", max_value, min_value,step
    # print "ddd"
    # print (max_value - min_value)/step

    for i in range(count_size):
        s = str(min_value + i * step) + ' < ' + str(min_value + (i + 1) * step)
        count_map[i] = 0
        show_map[i] = s

    for d in col_data:
        index = int((d - min_value) / step)
        if index < 0 or index >= 100:
            print "error", max_value, min_value, index, d
            raise NameError("max_value = %d min_value = min_value index 出错", max_value, min_value)
        count_map[index] = count_map[index] + 1

    for i in range(count_size):
        # print "^^^:",count_map[i],len(col_data)
        percent_map[i] = float(count_map[i] / float(len(col_data))) * 100
        # print percent_map[i]
    return show_map, count_map, percent_map


# @profile
def add_col_data(ws, col_data, global_min, global_max, name, max_x_width, min_, max_, dir_path):
    x_start = 1
    y_start = 0
    ws.write(0, 0, unicode("时间戳差", 'utf-8'))
    index = 0
    for i in col_data:
        if i < 0:
            raise NameError("错误数据")
        ws.write(x_start + index, 0, i)
        index = index + 1
    show_map, count_map, percent_map = get_timestamp_diff_statistics(col_data, global_min, global_max)
    ws.write(0, 1, unicode("时间戳差范围", 'utf-8'))
    ws.write(0, 2, unicode("次数统计", 'utf-8'))
    ws.write(0, 3, unicode("百分比", 'utf-8'))
    for i in range(count_size):
        ws.write(1 + i, 1, unicode(show_map[i], 'utf-8'))
        ws.write(1 + i, 2, count_map[i])
        ws.write(1 + i, 3, percent_map[i])
    # 添加 这2个值 好画图 统一坐标
    # col_data.append(global_min)
    # col_data.append(global_max)
    narray = np.array(col_data)
    file = generate_image_timestamp_diff(narray, name, max_x_width, dir_path, min_, max_)
    # covert_image_format(file)
    # ws.insert_image(name+'.bmp', 0, 6)
    ws.insert_image(12, 8, file)

    mean_v, var_v, std_v = get_mean_and_var(narray)
    ws.write(0, 4, unicode("均值", 'utf-8'))
    ws.write(0, 5, unicode("方差", 'utf-8'))
    ws.write(1, 4, mean_v)
    ws.write(1, 5, var_v)


def add_col_image(ws, col_data):
    pass


def get_max_min(data):
    global_min = sys.maxint
    global_max = 0
    for i in sorted(data.keys()):
        temp = filter(lambda x: x > 0, data[i])
        min_temp = min(temp)
        max_temp = max(temp)
        if global_min > min_temp:
            print i
            global_min = min_temp
        if global_max < max_temp:
            global_max = max_temp
        if global_min < 0:
            raise NameError("min 为负数 出错")

    # print global_min,global_max
    return global_min, global_max


def gd(x, m, s):
    left = 1 / (math.sqrt(2 * math.pi) * s)
    right = math.exp(-math.pow(x - m, 2) / (2 * math.pow(s, 2)))
    return left * right


# bws = .1
# pal = sns.blend_palette([sns.desaturate("royalblue", 0), "royalblue"], 5)
# plt.legend(title="kernel bandwidth value")
def generate_image_timestamp_diff(data, name, max_x_width, dir_path, min_=1, max_=3, image_type=None):
    # with Timer() as t:
    #    sns.kdeplot(data, bw=bws, color='#FF0000', lw=1.8, label=bws)
    #    #sns.rugplot(data, color="#CF3512")
    #    plt.xlim(min(data),min(data) + max_x_width)
    # print "diff", t.secs
    # bws = [.1]
    # for bw, c in zip(bws, pal):
    # sns.kdeplot(data, bw=bw, color=c, lw=1.8, label=bw)
    # plt.xlim(3000,4000)
    # plt.show()

    plt.figure(figsize=(11, 6))
    red_patch = mpatches.Patch(color='red', label=u'timestamp:%s' % name)
    # red_patch = mpatches.Patch(color='red', label=u'均值:方差:')
    red_patch = mpatches.Patch(color='red', label=u'avg:%s' % data.mean())
    blue_patch = mpatches.Patch(color='blue', label=u'std:%s' % data.std())
    green_patch = mpatches.Patch(color='green', label=u'var:%s' % data.var())
    plt.legend(handles=[red_patch, blue_patch, green_patch])

    plt.title(u"两个基站时间戳差%s分析" % name, fontproperties=zhfont1)
    plt.xlabel(u"时间戳差大小值", fontproperties=zhfont1)
    plt.ylabel(u"时间戳差值所占比重", fontproperties=zhfont1)
    plt.xlim(min(data), min(data) + max_x_width)
    # plt.xlim(0,data[index].max() + 0.1)
    # plt.savefig('percent.png', format='png')
    # plt.legend([red_dot,  white_cross], ["均值:", "方差:"])

    # if data[index].min() < 0:
    #    raise Exception('error distance data')
    # print type(data),data
    sns.kdeplot(data, color='#FF0000', shade=True, lw=1, legend=True)

    # 1 cm 精度
    mu, sigma = np.median(data), unit
    # mu, sigma = data[index].mean(),data[index].std()
    for i, c in zip([min_, max_], ['#00FF00', "#C1F320"]):
        s = np.random.normal(mu, float(i) * unit, 90000)
        sns.kdeplot(s, color=c, shade=True, lw=1, legend=True)

    # s = np.random.normal(mu, sigma, 90000)
    # sns.kdeplot(s,color='#00FF00', shade=True, lw=1,legend = True)
    # x = np.arange(data.min(),data.max(),0.001)
    # y=[]
    # for i in x:
    #    y.append(gd(i,0,1))
    # pl.plot(x,y)
    # y = stats.norm.pdf(x,( data.min()+data.max())/2,1)
    # mu, sigma = data.mean(), 1 # mean and standard deviation
    # s = np.random.normal(mu, sigma, 90000)
    # s = np.random.normal(data.min(),data.max(),size=90000)
    # sns.plot(x,y,color='#00FF00', shade=True, lw=1,legend = True)

    l = plt.axvline(data.mean(), color='red')
    l = plt.axvline(data.mean() + 1 * unit)
    l = plt.axvline(data.mean() - 1 * unit)
    l = plt.axvline(data.mean() + 2 * unit)
    l = plt.axvline(data.mean() - 2 * unit)
    # plt.show()
    if image_type == "png":
        plt.savefig(dir_path + '%s.png' % name, dpi=1000, format='png')
    else:
        plt.savefig(dir_path + '%s.png' % name, dpi=1000, format='png')
        plt.savefig(dir_path + '%s.svg' % name, format='svg', dpi=1200)
    plt.close()

    return name + '.png'


def get_max_range(data):
    result = 0
    for i in (sorted(data.keys())):
        filter_data = filter(lambda x: x > 0, data[i])
        max_i = max(filter_data)
        min_i = min(filter_data)
        if max_i - min_i > result:
            result = max_i - min_i
        if min_i < 0:
            raise Exception("数据错误")
    return result


def write_data(data, select_array, global_min, global_max, dir_path, min_, max_, count):
    # print select_array
    try:
        # print select_array
        max_x_width = get_max_range(data)
        # print max_x_width

        w = Workbook(dir_path + '/time_diff_result.xls')

        raw_data = w.add_worksheet("raw_data")
        col_size = len(data['table0_1'])
        for index, i in enumerate(sorted(data.keys())):
            raw_data.write(0, index, i)
            if col_size != len(data[i]):
                raise Exception("长度不一致")
            # print index ,i
            # filter_data = filter(lambda x:data[i].index(x) in select_array,data[i])
            # print len(filter_data)

            for inner_index, d in enumerate(data[i]):
                if inner_index in select_array:
                    # print inner_index
                    raw_data.write(inner_index + 1, index, d)
                else:
                    pass
                    # print inner_index,d

        std_map = {}
        for index, i in enumerate(sorted(data.keys())):
            print index, i
            ws = w.add_worksheet(i)
            # 去掉错误的数据 通过select_array过滤
            # print len(data[i])
            filter_data = filter(lambda x: data[i].index(x) in select_array, data[i])
            # print "type $$$$",type(filter_data)
            add_col_data(ws, filter_data, global_min, global_max, str(i), max_x_width, min_, max_, dir_path)
            # print type(data[i])
            # print "data[i]",data[i]
            # return
            std_map[index] = np.array(data[i]).std()

            # single_map[i] = np.array(data[i]).std()
            # print len(filter_data)
            # print type(data[i])
        # w.save('time_diff_result.xls')
        # return
        generate_timestamp_std_image_single_statsion_compare(std_map, dir_path, int(count))
        std_image = generate_timestamp_std_image(std_map, dir_path)
        raw_data.insert_image(0, 0, std_image)
        w.close()
    except Exception, e:
        print "excp", e
        traceback.print_exc()


def write_data_to_excel(file, dir_path, min_, max_, start, count):
    data, select_array, data_map = parse_timestamp_data(dir_path + 'aa.csv', dir_path, start, count)
    global_min, global_max = get_max_min(data)

    write_data(data, select_array, global_min, global_max, dir_path, min_, max_, count)
    generate_zipfile(dir_path)


def add_col_image(ws, col_data):
    pass


def test_time_profile():
    print "ddd"


test_single_map = {"table0_1": 81.912259345896288, "table0_2": 82.875007302170147, "table0_3": 81.823768813909496,
                   "table0_4": 76.38040460524212, "table0_5": 78.699921922535864, "table1_2": 79.943453806808506,
                   "table1_3": 84.884606687089573, "table1_4": 80.140806172528045, "table1_5": 80.71318828316393,
                   "table2_3": 86.913497520126384, "table2_4": 81.916965652346747, "table2_5": 82.074041210445955,
                   "table3_4": 80.977783639300426, "table3_5": 82.390622900645511, "table4_5": 72.585406460458657}

test_map = {0: 81.912259345896288, 1: 82.875007302170147, 2: 81.823768813909496, 3: 76.38040460524212,
            4: 78.699921922535864, 5: 79.943453806808506, 6: 84.884606687089573, 7: 80.140806172528045,
            8: 80.71318828316393, 9: 86.913497520126384, 10: 81.916965652346747, 11: 82.074041210445955,
            12: 80.977783639300426, 13: 82.390622900645511, 14: 72.585406460458657}


def generate_timestamp_std_image_single_statsion_compare(data, dir_path, count, image_type=None):
    li = generate_list_item(count)
    for i in xrange(count):
        plt.figure(figsize=(11, 6))
        x, y = [], []
        for d, key in enumerate(sorted(data)):
            if i in li[d]:
                for t in li[d]:
                    if t != i:
                        x.append(t)
                        y.append(data[key])
        plt.plot(x, y, 'ro')
        plt.margins(0.05)
        plt.title(u"以%s号基站为基准的时间戳差分析" % i, fontproperties=zhfont1)
        plt.xlabel(u"另外一个基站编号", fontproperties=zhfont1)
        plt.ylabel(u"时间戳差", fontproperties=zhfont1)
        if image_type == "png":
            plt.savefig(dir_path + 'single%s_diff_std.png' % i, dpi=1000, format='png')
        else:
            plt.savefig(dir_path + 'single%s_diff_std.png' % i, dpi=1000, format='png')
            plt.savefig(dir_path + 'single%s_diff_std.svg' % i, format='svg', dpi=1200)
        plt.close()
        # plt.show()


def generate_timestamp_std_image(map_data, dir_path, image_type=None):
    plt.figure(figsize=(11, 6))
    x, y = [], []
    print map_data
    for key in map_data:
        x.append(key)
        y.append(map_data[key])
    plt.plot(x, y, 'ro')
    plt.margins(0.05)
    plt.title(u"两个基站时间戳差分析系统测距误差范围", fontproperties=zhfont1)
    plt.xlabel(u"两个基站对比编号", fontproperties=zhfont1)
    plt.ylabel(u"时间戳差", fontproperties=zhfont1)
    l = plt.axhline(y=np.array(y).mean(), linewidth=2, color='b')
    # l = plt.axhline(y=np.array(y).mean() + unit,linewidth=2, color='g')
    # l = plt.axhline(y=np.array(y).mean() - unit,linewidth=2, color='r')
    # l.set_label("test")
    # plt.show()
    if image_type == "png":
        plt.savefig(dir_path + 'timestamp_diff_std.png', dpi=1000, format='png')
    else:
        plt.savefig(dir_path + 'timestamp_diff_std.png', dpi=1000, format='png')
        plt.savefig(dir_path + 'timestamp_diff_std.svg', format='svg', dpi=1200)
    plt.close()
    return 'timestamp_diff_std.png'


def write_data(f, start, count, num):
    result = {}
    base_num = int(num)
    start_para = int(start)
    end_para = int(start) + int(count)
    col = range(max_value)
    use_cols = col[start_para:end_para]
    df = pd.read_csv(f, usecols=use_cols, header=None)
    # 下面2行 由于保留小数点后面三位 整体转换为int类型 方便计算
    float_data = df.applymap(lambda x: x * 1000)
    data = float_data.astype('int64')
    for i in col[start_para + 1:end_para]:
        count_all = 0
        diff_count = 0
        diff = two_list_sub(data[i].tolist(), data[start_para])
        for index in xrange(len(diff) - 1):
            diff_value = diff[index + 1] - diff[index]
            count_all = count_all + 1
            if abs(diff_value) < 200:
                diff_count = diff_count + 1
        # print count_all,diff_count
        result[i] = float(diff_count) / float(count_all)
        # print result[i]
    # print result
    return result


# index_li = [u'1',u'2', u'4', u'5', u'3']
def cut_unicode(li):
    result = []
    for i in li:
        ii = str(i)
        result.append(ii)
    return result


@exeTime
def process3(f, index_li, start):
    # bool_all_dic = {}
    pattern = {}
    rate_pattern = {}
    index_li = cut_unicode(index_li)
    cut_li = list(index_li)
    lines = getLineCount(f)
    input_dic = {}
    for i in index_li:
        input_dic[i] = [0] * lines
    # print input_dic
    data = pd.DataFrame(input_dic)
    # print data
    line_no = 0
    with open(f) as file:
        for line in file:
            # print line_no
            line = line.strip('\r\n')
            temp = re.split(",", line)
            temp_dic = {}
            for i, e in enumerate(temp[0:20]):
                # for i,e in enumerate(temp[0:20]):
                if e != '0':
                    # print i ,e
                    temp_dic[unicode(e, "utf-8")] = i
                    if e in cut_li:
                        cut_li.remove(e)
                        # bool_all_dic[unicode(e,"utf-8")] = True
            # print temp_dic.keys()
            if frozenset(temp_dic.keys()) in pattern.keys():
                pattern[frozenset(temp_dic.keys())] = pattern[frozenset(temp_dic.keys())] + 1
            else:
                pattern[frozenset(temp_dic.keys())] = 1
            # print temp_dic
            # print data.loc[line_no]
            for i in temp_dic.keys():
                # print i,data.loc[line_no][i]
                # print temp[21 + temp_dic[i]]
                data.loc[line_no][i] = int(float(temp[21 + temp_dic[i]]) * 1000)
                # data.loc[line_no][i] = int( float(temp[21 + temp_dic[i]]) * 1000)
            line_no = line_no + 1
            # print data.loc[line_no]
    for e in pattern.keys():
        rate_pattern[e] = float(pattern[e]) / float(lines)

    # 选择多个基准站
    detail_result = {}

    print cut_li
    for i in cut_li:
        del data[i]
    print data.columns
    print data

    result = {}
    miss_result = {}
    for i in list(data.columns):
        print "column", i ,type(i)
        count_all = 0
        diff_count = 0
        miss_count = 0
        temp_list = []
        # print type(i),i
        if i == start:
            print "continue"
            continue
        for index, row in data.iterrows():
            # print index, row
            if row[i] == 0:
                miss_count = miss_count + 1
                continue
            diff_value = int(row[i] - row[start])
            temp_list.append(diff_value)
            count_all = count_all + 1
        for index, e in enumerate(temp_list):
            if index == 0:
                continue
            # print temp_list[index] ,temp_list[index - 1]
            temp_diff = int(temp_list[index]) - int(temp_list[index - 1])
            if temp_diff > -200 and temp_diff < 200:
                diff_count = diff_count + 1
            else:
                pass
                # print "haha"

        # diff = two_list_sub(data[i].tolist(),data[start])
        # for index in xrange(len(diff) - 1):
        #    count_all = count_all + 1
        #    if abs(diff_value) < 200:
        #        diff_count = diff_count + 1
        ##print count_all,diff_count
        # print diff_count,count_all
        result[i] = float(diff_count) / float(count_all)
        miss_result[i] = float(miss_count) / float(lines)
        # print result[i]
    # print result
    # print all_dic
    # print rate_pattern,pattern
    print index_li
    print cut_li
    none_data_columns = list(set(index_li).difference(set(cut_li)))
    print rate_pattern, pattern, result, miss_result, cut_li
    return rate_pattern, pattern, result, miss_result, cut_li

def process4(f,index_li):
    # bool_all_dic = {}
    print "4process"
    rate_pattern = {}
    #index_li = cut_unicode(index_li)

    lines = getLineCount(f)
    input_dic = {}
    for i in index_li:
        input_dic[i] = [0] * lines
    # print input_dic
    data = pd.DataFrame(input_dic)
    # print data
    line_no = 0
    with open(f) as file:
        for line in file:
            # print line_no
            line = line.strip('\r\n')
            temp = re.split(",", line)
            temp_dic = {}
            for i, e in enumerate(temp[0:20]):
                # for i,e in enumerate(temp[0:20]):
                if e != '0':
                    # print i ,e
                    temp_dic[unicode(e, "utf-8")] = i
            # print temp_dic
            # print data.loc[line_no]
            for i in temp_dic.keys():
                # print i,data.loc[line_no][i]
                # print temp[21 + temp_dic[i]]
                data.loc[line_no][i] = int(float(temp[21 + temp_dic[i]]) * 1000)
                # data.loc[line_no][i] = int( float(temp[21 + temp_dic[i]]) * 1000)
            line_no = line_no + 1


    count_map = {}

    count_map_detail = {}
    count_map_result = {}
    # start_para 只是为了行遍历
    for i in range(data[index_li[0]].size):
        temp_map = {}
        for j in index_li:
            # temp_map 前面是时间差 后面是基站序号
            # {239421893: 11, 239420911: 8, 239417008: 12, 239420273: 9, 239422645: 10, 239421721: 7}
            temp_map[data[j][i]] = j
            # print data[j][i]
        # dict= sorted(map.items(), key=lambda d:d[0], reverse = False)
        # index_map 是时间差排序序号的一个列表 如果存在 计数+1 否则创建1
        index_map = sortedDictValues(temp_map)
        #print temp_map,index_map
        if str(index_map) in count_map.keys():
            count_map[str(index_map)] = count_map[str(index_map)] + 1
        else:
            print "hhah"
            count_map[str(index_map)] = 1
            # 时间差排序列表 对应那些行在这个序列里面
            count_map_detail[str(index_map)] = []
            #count_map_result[str(index_map)] = index_map
        count_map_detail[str(index_map)].append(i)
    print count_map_detail
    #print count_map_result
    data_map = sorted(count_map.items(), key=lambda d: d[1],reverse=True)[0][0]
    print data_map
    data_map_index = sorted(count_map.items(), key=lambda d: d[1], reverse=True)[0][0]
    b = xrange(line_no)

    data =  data.drop(data.index[list(set(b).difference(set(count_map_detail[str(index_map)])))])
    print data
    var_result = {}
    result_compare = generate_compare(index_li)
    for i in result_compare:
        #print type(data[i[0]] - data[i[1]])
        #print (data[i[0]] - data[i[1]]).std()
        var_result[i] = (data[i[0]] - data[i[1]]).std()
    print var_result
    return str(var_result)


#index_li = [11, 12, 13, 14, 21, 23, 24]
# start = '1'
#process3('./test.csv', index_li, '11')


# write_data("./Record.csv",7,6)
# parse_timestamp_data("./Record.csv","dd")
# generate_timestamp_std_image_single_statsion_compare(test_map,"./",6)

# generate_timestamp_std_image(test_map)
# generate_list_item()
# try:
##    with Timer() as t:
##        test_time_profile()
#    #test_time_profile()
# write_data_to_excel('./Record.csv','./')
# except Exception, e:
#    print "excp", e
#    traceback.print_exc()

# 每次按时间差排序 排序列表不存在
