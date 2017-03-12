import xmltodict
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
import cPickle as pkl
# import json

dst = pkl.load(open('pd_direction.pkl', 'rb'))

# distance data
t = list(dst.index)
p1 = dst.iloc[:, 0]
P = dst.iloc[:, :]

print t
print p1

# visualize minimum distance to ball
title = 'direction difference between players and ball'
plt.title(title)
plt.xlabel('Time[sec]')
plt.ylabel('direction difference[deg]')

# revert x axis
plt.plot(t, P, label='player')
plt.grid(True)
#plt.legend()
#plt.ylim(0, 5)
plt.show()

## ball data
#    ball = []
#    b_xs = []
#    b_ys = []
#    b_zs = []
#    #    for i in range(num):
#    for i in range(1000):
#        locs = datatraj[i]['@locations']
#        locs = locs.split(';')
#        print locs[0]
#        b_xs.append(float(locs[0].split(',')[-3]))
#        b_ys.append(float(locs[0].split(',')[-2]))
#        b_zs.append(float(locs[0].split(',')[-1]))
#
#    # visualize minimum distance to ball
#    title = 'ball to players distance'
#    plt.title(title)
#    plt.xlabel('Time[sec]')
#    plt.ylabel('Scaled')
#    # plt.xlim(0,0.01)
#    # plt.ylim(0, 100)
#    plt.grid(True)
#    plt.plot(t, s_v, label='s_velocity')
#    plt.plot(t, s_b, label='s_brake')
#    plt.plot(t, s_th, label='s_throttle')
#    plt.plot(t, s_e, label='s_ev')
#    plt.plot(t, s_f, label='s_fuel')
#    plt.plot(t, s_c, label='s_charge')
#    plt.plot(t, s_z, label='s_altitude')
#    plt.legend()
