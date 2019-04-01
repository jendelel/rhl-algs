from PyQt5 import QtWidgets, QtCore, QtGui
import qimage2ndarray
from utils import seconds_to_text
import time
import sys


class MyPygletViewer(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.verLayout = QtWidgets.QVBoxLayout()
        self.img = QtWidgets.QLabel()
        self.img.setMinimumSize(1, 1)
        self.img.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.infoLabel = QtWidgets.QLabel()
        self.infoLabel.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.infoLabel.setText("Information")
        self.verLayout.addWidget(self.img)
        self.verLayout.addWidget(self.infoLabel)
        self.setLayout(self.verLayout)

        self.start_time = None
        self.episode_count = 0
        self.reward = 0
        self.action = 0
        self.acc_reward = 0
        self.step_count = 0
        self.num_pos_feedback = 0
        self.num_neg_feedback = 0

    def imshow(self, img):
        qimg = qimage2ndarray.array2qimage(img, True).scaled(self.img.size())
        self.img.setPixmap(QtGui.QPixmap(qimg))
        self.update_info()
        self.repaint()

    def update_info(self):
        time_str = ""
        if self.start_time:
            time_str = seconds_to_text(time.time() - self.start_time)
        self.infoLabel.setText("Step: {}, Episode: {}, Action: {}, Time: {}, Reward: {:.2f}, "
                               "Acc_reward: {:.2f}, pos_feedback: {}, neg_feedback: {}".format(
                                       self.step_count, self.episode_count, self.action, time_str, self.reward,
                                       self.acc_reward, self.num_pos_feedback, self.num_neg_feedback))


class MainWindow(QtWidgets.QMainWindow):
    keyPressedSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        # Agent scene.
        self.viewer = MyPygletViewer()
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        ver = QtWidgets.QVBoxLayout()
        ver.addWidget(self.viewer)

        # Algorithm specific buttons (feedback)
        self.feedback_widget = QtWidgets.QWidget()
        self.feedback_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        ver.addWidget(self.feedback_widget)

        # General controls (Start, Record, Information...)
        hor = QtWidgets.QHBoxLayout()
        self.recordCheck = QtWidgets.QCheckBox()
        self.recordCheck.setChecked(False)
        self.recordCheck.setText("Record")
        hor.addWidget(self.recordCheck)

        self.trainCheck = QtWidgets.QCheckBox()
        self.trainCheck.setChecked(True)
        self.trainCheck.setText("Train")
        hor.addWidget(self.trainCheck)

        spinWrapper = QtWidgets.QHBoxLayout()
        spinLabel = QtWidgets.QLabel()
        spinLabel.setText("Render sleep:")
        spinWrapper.addWidget(spinLabel)
        self.renderSpin = QtWidgets.QDoubleSpinBox()
        self.renderSpin.setRange(0.0005, 1)
        self.renderSpin.setSingleStep(0.01)
        self.renderSpin.setValue(0.02)
        spinWrapper.addWidget(self.renderSpin)
        hor.addLayout(spinWrapper)

        self.startBut = QtWidgets.QPushButton()
        self.startBut.setText("Start!")
        hor.addWidget(self.startBut)
        ver.addLayout(hor)

        # Putting everything together.
        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(ver)
        self.setCentralWidget(centralWidget)
        self.resize(QtWidgets.QDesktopWidget().availableGeometry(self).size() * 0.7)
        self.bringToFront()  # Also sets to be always on top.

    def render(self, env):
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents)
        env.render('human')

    def processEvents(self):
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
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.raise_()  # for MacOS
        self.activateWindow()  # for Windows


def create_app():
    return QtWidgets.QApplication(sys.argv)
