import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QGridLayout,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *



class FastImageWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None

        self.refresh_timer = QTimer()

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex_id = glGenTextures(1)

    def paintGL(self):
        # glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.image is None:
            return

        h, w, _ = self.image.shape

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            w, h, 0,
            GL_RGB, GL_UNSIGNED_BYTE,
            self.image
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()

    def update_image(self, image: np.ndarray):
        self.image = image
        self.update()
        # self.repaint()

from viewer_point_cloud import PointCloudWidget, PointCloudViewer
from viewer_gaussian import GaussianViewer

class MainViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("4-View PyQt Window")
        self.resize(1600, 1200)

        central = QWidget(self)
        self.setCentralWidget(central)

        grid = QGridLayout(central)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.setSpacing(5)

        # 四个区域
        self.view1 = PointCloudViewer()
        self.view2 = GaussianViewer()
        self.view3 = FastImageWidget()
        self.view4 = FastImageWidget()

        grid.addWidget(self.view1, 0, 0)
        grid.addWidget(self.view2, 0, 1)
        grid.addWidget(self.view3, 1, 0)
        grid.addWidget(self.view4, 1, 1)

        grid.setRowStretch(0, 1)  # 第0行占1份空间
        grid.setRowStretch(1, 1)  # 第1行占1份空间
        grid.setColumnStretch(0, 1)  # 第0列占1份空间
        grid.setColumnStretch(1, 1)  # 第1列占1份空间

        
        # 示例：给每个区域放一张图片（可删）
        # self.view1.set_image("img1.jpg")
        # self.view2.set_image("img2.jpg")
        # self.view3.set_image("img3.jpg")
        # self.view4.set_image("img4.jpg")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainViewerWindow()
    window.show()
    import cv2
    img = cv2.imread("/home/xjc/code/livo2gs/gaussian-splatting/img1.png")
    img = np.array(img, dtype=np.uint8)
    # window.view2.update_image(img)
    window.view3.update_image(img)
    window.view4.update_image(img)

    sys.exit(app.exec_())
