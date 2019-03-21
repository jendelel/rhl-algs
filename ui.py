from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL
import qimage2ndarray

import sys

class MyPygletViewer(QtWidgets.QLabel):
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        # self.setMinimumSize(QtCore.QSize(480, 640))

    def imshow(self, img):
        qimg = qimage2ndarray.array2qimage(img, True).scaled(self.size())
        self.setPixmap(QtGui.QPixmap(qimg))
        self.repaint()

class MainWindow(QtWidgets.QMainWindow):
    keyPressedSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
              
        self.viewer = MyPygletViewer()
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        hor = QtWidgets.QHBoxLayout()
        self.startBut = QtWidgets.QPushButton()
        self.startBut.setText("Start!")

        hor.addWidget(self.startBut)
        ver = QtWidgets.QVBoxLayout()
        ver.addWidget(self.viewer)
        self.feedback_widget = QtWidgets.QWidget()
        self.feedback_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        ver.addWidget(self.feedback_widget)
        ver.addLayout(hor)
        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(ver)
        self.setCentralWidget(centralWidget)

        # self.startBut.clicked.connect(self.startRl)
        self.setFixedSize(QtCore.QSize(640, 480))
        self.bringToFront()
        self.running = False
    
    def render(self, env):
        env.render('human')
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents)

    def loadAlg(self, args_parser):
        argv = [str(arg) for arg in QtCore.QCoreApplication.instance().arguments()]
        args, _ = args_parser.parse_known_args(argv[1:])
        alg_name = args.alg
        module = __import__(alg_name)
        module.parse_args(args_parser)
        args = args_parser.parse_args(argv[1:])
        return module, args

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            event.accept()
            self.keyPressedSignal.emit(event)
        else:
            event.ignore()

    def bringToFront(self):
        self.setWindowState( (self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.raise_()  # for MacOS
        self.activateWindow() #  for Windows

def create_app():
    return QtWidgets.QApplication(sys.argv)