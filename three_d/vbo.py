from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np

WIN_W, WIN_H = 640, 640
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 0.5, 50.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE = np.array([0.1, 1.0, 1.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
LEFT_IS_DOWNED = False  # 鼠标左键被按下
scope = 1

# 顶点集
# vertices = np.array([
#     -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5,  # v0-v1-v2-v3
#     -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5  # v4-v5-v6-v7
# ], dtype=np.float32)

vertices = np.array([
    0.3, 0.6, 0.9, -0.35, 0.35, 0.35,  # c0-v0
    0.6, 0.9, 0.3, 0.35, 0.35, 0.35,  # c1-v1
    0.9, 0.3, 0.6, 0.35, -0.35, 0.35,  # c2-v2
    0.3, 0.9, 0.6, -0.35, -0.35, 0.35,  # c3-v3
    0.6, 0.3, 0.9, -0.35, 0.35, -0.35,  # c4-v4
    0.9, 0.6, 0.3, 0.35, 0.35, -0.35,  # c5-v5
    0.3, 0.9, 0.9, 0.35, -0.35, -0.35,  # c6-v6
    0.9, 0.9, 0.3, -0.35, -0.35, -0.35  # c7-v7
], dtype=np.float32)

# 索引集
indices = np.array([
    0, 1, 2, 3,  # v0-v1-v2-v3 (front)
    4, 5, 1, 0,  # v4-v5-v1-v0 (top)
    3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
    5, 4, 7, 6,  # v5-v4-v7-v6 (back)
    1, 5, 6, 2,  # v1-v5-v6-v2 (right)
    4, 0, 3, 7  # v4-v0-v3-v7 (left)
], dtype=np.int)

vbo_vertices = vbo.VBO(vertices)
vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)


def getposture():
    global EYE, LOOK_AT

    dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
        theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0

    return dist, phi, theta


DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


def start_draw():
    vbo_vertices.bind()
    # glInterleavedArrays(GL_V3F, 0, None)
    glInterleavedArrays(GL_C3F_V3F, 0, None)
    vbo_vertices.unbind()
    vbo_indices.bind()
    glDrawElements(GL_QUADS, int(vbo_indices.size / 4), GL_UNSIGNED_INT, None)
    vbo_indices.unbind()



def draw():
    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4],
              VIEW[5])  # glFrustum() 用来设置平行投影

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

    # ---------------------------------------------------------------
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-scope, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(scope, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）
    glVertex3f(scope, 0.0, 0.0)  # x轴正方向箭头
    glVertex3f(scope - 0.1 * scope, -0.05 * scope, 0.0)  # x轴正方向箭头

    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -scope, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, scope, 0.0)  # 设置y轴顶点（y轴正方向）
    glVertex3f(0.0, scope, 0.0)  # y轴正方向箭头
    glVertex3f(-0.05 * scope, scope - 0.1 * scope, 0.0)  # y轴正方向箭头

    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -scope)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, scope)  # 设置z轴顶点（z轴正方向）
    glVertex3f(0.0, 0.0, scope)  # z轴正方向箭头
    glVertex3f(0.0, -0.05 * scope, scope - 0.1 * scope)  # z轴正方向箭头

    glEnd()  # 结束绘制线段

    # our_font.glPrint(320, 320, u"XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    start_draw()
    # ---------------------------------------------------------------
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y

    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H

    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y

        PHI += 2 * np.pi * dy / WIN_H
        PHI %= 2 * np.pi
        THETA += 2 * np.pi * dx / WIN_W
        THETA %= 2 * np.pi
        r = DIST * np.cos(PHI)

        EYE[1] = DIST * np.sin(PHI) * scope
        EYE[0] = r * np.sin(THETA) * scope
        EYE[2] = r * np.cos(THETA) * scope

        if 0.5 * np.pi < PHI < 1.5 * np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0
        glutPostRedisplay()


if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL'.encode("gbk"))

    init()  # 初始化画布
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    glutMainLoop()  # 进入glut主循环
