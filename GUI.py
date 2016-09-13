#!/usr/bin/python
# -'''- coding: utf-8 -'''-
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtWebKit import *
from time_tool import *

from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('./'))
# Capture our current directory

codec = QTextCodec.codecForName("utf-8")
class Special_Widget(QWidget):
    def __init__(self, parent=None):
        super(Special_Widget, self).__init__()

        self.label1 = QLabel(codec.toUnicode("指定基准基站号"))
        self.baseinput = QTextEdit()
        self.label2 = QLabel(codec.toUnicode("所有基站号"))
        self.all_bs = QTextEdit()
        self.filebutton = QPushButton(codec.toUnicode("选择要分析的文件"))

        self.leftPanel = QWidget()
        self.leftPanelLayout = QVBoxLayout()
        self.leftPanelLayout.addWidget(self.label1)
        self.leftPanelLayout.addWidget(self.baseinput)
        self.leftPanelLayout.addWidget(self.label2)
        self.leftPanelLayout.addWidget(self.all_bs)
        self.leftPanelLayout.addWidget(self.filebutton)
        self.leftPanel.setLayout(self.leftPanelLayout)

        self.connect(self.filebutton, SIGNAL("clicked()"),
                     self, SLOT("file_choose()"))
        #self.result = QTextEdit()
        self.result = QWebView()
        self.result.setUrl("./welcome.html")
        #self.result.load(QUrl("http://www.baidu.com"));

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.leftPanel)
        self.layout.addWidget(self.result)
        self.layout.setStretchFactor(self.leftPanel,1)
        self.layout.setStretchFactor(self.result, 6)
        self.setLayout(self.layout)

    def file_choose(self):
        try:
            fname = QFileDialog.getOpenFileName()
            print fname[0]
            print str(self.baseinput.toPlainText())
            print str(self.all_bs.toPlainText())
            index_li = re.split('[!?, ]', str(self.all_bs.toPlainText()))
            #index_li = [11,21,31,41]
            start = str(self.baseinput.toPlainText())
            self.result.setUrl("./processing.html")
            rate_pattern, pattern, result, miss_result, none_data_columns = process3(fname[0],index_li,start)
            template = env.get_template('timestamp_result3.html')
            output_from_parsed_template = template.render(result=result,
                                                          rate_pattern=rate_pattern,
                                                          miss_result = miss_result,
                                                          none_data_columns = none_data_columns)
            self.result.setHtml(output_from_parsed_template)
        except Exception,e:
            print e
            traceback.print_exc

class Non_Special_Widget(QWidget):
    def __init__(self, parent=None):
        super(Non_Special_Widget, self).__init__()
        self.label2 = QLabel(codec.toUnicode("所有基站号"))
        self.all_bs = QTextEdit()
        self.filebutton = QPushButton(codec.toUnicode("选择要分析的文件"))
        self.leftPanel = QWidget()
        self.leftPanelLayout = QVBoxLayout()
        self.leftPanelLayout.addWidget(self.label2)
        self.leftPanelLayout.addWidget(self.all_bs)
        self.leftPanelLayout.addWidget(self.filebutton)
        self.leftPanel.setLayout(self.leftPanelLayout)

        self.connect(self.filebutton, SIGNAL("clicked()"),
                     self, SLOT("file_choose()"))
        self.result = QTextEdit()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.leftPanel)
        self.layout.addWidget(self.result)
        self.layout.setStretchFactor(self.leftPanel,1)
        self.layout.setStretchFactor(self.result, 6)
        self.setLayout(self.layout)


    def file_choose(self):
        try:
            fname = QFileDialog.getOpenFileName()
            print fname[0]
            index_li = re.split('[!?, ]', str(self.all_bs.toPlainText()))
            self.result.setText(process4(fname[0], index_li))
        except Exception, e:
            print e
            traceback.print_exc

class DataWidget(QTabWidget):
    def __init__(self, parent=None):
        super(DataWidget, self).__init__()

        self.special_Widget = Special_Widget()
        self.addTab(self.special_Widget, codec.toUnicode("指定基站方式"))
        self.non_Special_Widget = Non_Special_Widget()
        self.addTab(self.non_Special_Widget, codec.toUnicode("不指定基站方式"))


class GateWidget(QWidget):
    def __init__(self, parent=None):
        super(GateWidget, self).__init__()
        #super(GateWidget, self).__init__(parent)


class GUI(QTabWidget):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)
        self.resize(800, 600)
        self.codec = QTextCodec.codecForName("utf-8")

        #self.tabWidget.setBackgroundRole(Qt.red)
        self.gateWidget = GateWidget()
        self.addTab(self.gateWidget, codec.toUnicode("门限分析"))
        self.dataWidget = DataWidget()
        self.addTab(self.dataWidget, codec.toUnicode("时间戳分析"))

        #self.add

        #
        # # self.rightLayout = QVBoxLayout()
        # # self.rightLayout.addChildWidget(tabWidget)
        # # self.lefLayout = QVBoxLayout()
        #
        # self.layout = QHBoxLayout()
        # self.layout.addChildWidget(self.leftWidget)
        # self.layout.addChildWidget(self.tabWidget)

        #layout.setStretchFactor(self.lefLayout, 7)
        #layout.setStretchFactor(self.rightLayout, 1)
        # Set dialog layout
        #self.setLayout(self.layout)
