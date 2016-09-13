#!/usr/bin/python
# -'''- coding: utf-8 -'''-

import sys
from PySide.QtCore import *
from PySide.QtGui import *
from GUI import *

if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    gui = GUI()
    gui.show()
    # Run the main Qt loop
    sys.exit(app.exec_())