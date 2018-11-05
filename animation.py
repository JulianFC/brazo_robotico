"""
============
3D animation
============

A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from denavit_hartenberg import direct_problem
from math import pi




def update_lines(num, QT, lines, end):
    print(end)
    QT,P = direct_problem([end[0]*num/200, end[1]*num/200, end[2]*num/200, end[3]*num/200], 4)
    for line in range(4):
        lines[line].set_data(QT[0:2, line:line+2])
        lines[line].set_3d_properties(QT[2, line:line+2])
    #for line, data in zip(lines, dataLines):
    #    # NOTE: there is no .set_data() for 3 dim data...
    #    line.set_data(data[0:2, :num])
    #    line.set_3d_properties(data[2, :num])
    #print(P)
    return lines


def animate(start, end):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Fifty lines of random 3-D lines
    #data = [Gen_RandLine(25, 3) for index in range(50)]
    N = 4
    QT, P = direct_problem(start, N)

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(QT[0, i:i + 2],  QT[1, i:i + 2], QT[2, i:i + 2], linewidth=6)[0] for i in range(4)]

    # Setting the axes properties
    ax.set_xlim3d([-3.0, 3.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-3.0, 3.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-3.0, 3.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines,  frames=200, fargs=(QT, lines, end),
                                       interval=10, blit=True,repeat=False)

    plt.show()
