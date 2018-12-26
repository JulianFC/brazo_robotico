import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




def direct_problem(Theta,N): #,fig,ax):
    L1 = 1
    L2 = 1
    L3 = 1
    L4 = 1

    theta1 = Theta[0]
    theta2 = Theta[1]
    theta3 = Theta[2]
    theta4 = Theta[3]

    QT = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype='f')


    R1 = np.array([[cos(theta1), 0, sin(theta1), 0], [sin(theta1), 0, -cos(theta1), 0], [0, 1, 0, L1], [0, 0, 0, 1]])
    R2 = np.array([[cos(theta2), -sin(theta2), 0, L2 * cos(theta2)], [sin(theta2), cos(theta2), 0, L2 * sin(theta2)], [0, 0, 1, 0], [0, 0, 0, 1]])
    R3 = np.array([[cos(theta3), -sin(theta3), 0, L3 * cos(theta3)], [sin(theta3), cos(theta3), 0, L3 * sin(theta3)], [0, 0, 1, 0], [0, 0, 0, 1]])
    R4 = np.array([[cos(theta4), -sin(theta4), 0, L4 * cos(theta4)], [sin(theta4), cos(theta4), 0, L4 * sin(theta4)], [0, 0, 1, 0], [0, 0, 0, 1]])

    Q1 = np.matmul(R1, QT[:, 1])
    Q2 = np.matmul(np.matmul(R1, R2), QT[:, 2])
    Q3 = np.matmul(np.matmul(np.matmul(R1, R2), R3), QT[:, 3])
    Q4 = np.matmul(np.matmul(np.matmul(np.matmul(R1, R2), R3), R4), QT[:, 4])


    QT[:, 1] = Q1
    QT[:, 2] = Q2
    QT[:, 3] = Q3
    QT[:, 4] = Q4

    #P = plot_arm(QT, N,fig,ax)
    P = np.array([QT[0, 4], QT[1, 4], QT[2, 4]])
    return QT, P


def plot_arm(QT, N,fig,ax):
    P = np.array([QT[0, 4], QT[1, 4], QT[2, 4]])
    #for i in range(N):
        #ax.plot3D(QT[0, i:i + 2], QT[1, i:i + 2], QT[2, i:i + 2], linewidth=6)
    #plt.show()
    #ax.plot3D(QT[0, 0:5], QT[1, 0:5], QT[2, 0:5], linewidth=6)
    #plt.show()
    return P


#fig = plt.figure()
#ax = plt.axes(projection='3d')
#plt.ylim(0, 3)
#plt.xlim(0, 3)
#ax.set_zlim(0, 2)

#QT, P = direct_problem([pi/4, pi/4, -pi/4, -pi/4], 4, fig, ax)
