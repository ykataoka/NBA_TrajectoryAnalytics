#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xmltodict
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
import cPickle as pkl
from scipy import stats
import numpy as np

# import json

if __name__ == '__main__':
    """
    read trajectory data
    """
    # read trajectory data
    filetraj = 'NBA_FINAL_SEQUENCE_OPTICAL$2016012523_Q1.XML'
    str = open(filetraj, 'r').read()
    result = xmltodict.parse(str)

    # parse to dict
    data = dict(result)
    datatraj = data['sports-statistics']['sports-boxscores']['nba-boxscores']['nba-boxscore']['sequences']['moment']
    num = len(datatraj)

    # make playerID's list for both team
    # {team1:set(1,3,5,7,9,11,13...), team2:set(0,2,4,6,8,10,12...)}
    team1_players = set()
    team2_players = set()
    for i in range(num):
        locs = datatraj[i]['@locations']
        locs = locs.split(';')
        for loc in locs:
            loc_list = loc.split(',')
            if loc_list[0] == '-1':
                continue
            elif loc_list[0] == '23':
                team1_players.update([loc_list[1]])
            elif loc_list[0] == '5312':
                team2_players.update([loc_list[1]])

    """
    Distance
    """
# dumping distance data is DONE!
#    """
#    compute distance between ball and players (dist)
#    """
#    # set the pandas data structure to save distance ball (dist)
#    # time(idx), playerID1, playerID2, ..., playerID 17
#    cols1 = list(team1_players)
#    cols2 = list(team2_players)
#    cols = cols1 + cols2
#    col_num = len(team1_players) + len(team2_players)
#    dist = pd.DataFrame([[None]*col_num], columns=cols, index=[720.0])
#    print 'number of trajectory data -> ', num
#
#    # compute the dist by all the data
#    for i in range(num):
#        print i, '/', num, '\r'
#        # get game time
#        timeNow = float(datatraj[i]['@game-clock'])
#
#        # get locations
#        locs = datatraj[i]['@locations']
#        locs = locs.split(';')
#        ballX, ballY = 0, 0
#        tmp = pd.DataFrame([[None]*col_num], columns=cols, index=[timeNow])
#        for loc in locs:  # loop 11 datas
#            loc_list = loc.split(',')  # [23,262882,20.47625,21.10303,0]
#            if loc_list[0] == '-1':  # ball
#                ballX = float(loc_list[2])
#                ballY = float(loc_list[3])
#            else:  # player's trajectory
#                playerID = loc_list[1]
#                playerX = float(loc_list[2])
#                playerY = float(loc_list[3])
#                dist_tmp = math.sqrt(pow((ballX - playerX), 2) +
#                                     pow((ballY - playerY), 2))
#                tmp.ix[timeNow, playerID] = dist_tmp  # update distance value
#
#        # update the dist
#        dist = dist.append(tmp)
#
#    # save to pickle
#    pkl.dump(dist, open('pd_distance.pkl', 'wb'))


    """
    Direction Difference
    """
## dumping degree difference data is DONE!
#    """
#    compute direction similarity between ball and players (dirc)
#    """
#    # read x,y coordinate to pandas
#    Xs = pkl.load(open('pd_Xs.pkl', 'rb'))
#    Ys = pkl.load(open('pd_Ys.pkl', 'rb'))
#
#    cols1 = list(team1_players)
#    cols2 = list(team2_players)
#    cols = cols1 + cols2
#    col_num = len(team1_players) + len(team2_players)
#    dirc = pd.DataFrame([[None]*col_num], columns=cols, index=[720.0])
#
#    # loop for every data (direction is computed by 1 sec data)
#    for i in range(len(Xs)-25):
#        print i, '/', len(Xs)-25
#        if i == 0:
#            continue  # skip the bug for now...
#        
#        X = Xs.iloc[i:i+25, :]
#        Y = Ys.iloc[i:i+25, :]
#        X = X.dropna(axis=1)
#        Y = Y.dropna(axis=1)
#
#        timeNow = X.index[-1]
#        col_num = len(X.columns)-1
#        cols = X.columns[1:]
#        tmp = pd.DataFrame([[None]*col_num], columns=cols, index=[timeNow])
#        slope_deg_ball = 0.0  # initialize
#        for col in X.columns:
#            x = X.ix[:, col]
#            y = Y.ix[:, col]
#
#            # compute the slope
#            slope, intercept, r_value, _, _ = stats.linregress(x, y)
#            if col == 'ball':
#                slope_deg_ball = np.arctan(slope) * 180.0 / math.pi  # [deg]
#            else:
#                slope_deg = np.arctan(slope) * 180.0 / math.pi  # [deg]
#
#                # store to the data
#                slope_diff = abs(slope_deg - slope_deg_ball)
#                tmp.ix[timeNow, col] = slope_diff
#
#        # update the dirc
#        dirc = dirc.append(tmp)
#
#    # save to pickle
#    pkl.dump(dirc, open('pd_direction.pkl', 'wb'))

    """
    Derivative of the Distance
    """
#    # dumping derivative of the distance(velocity) is DONE!
#    """
#    compute derivative of the distance between ball and players (ddt_dist)
#    """
#    # set the pandas data structure to save distance ball (dist)
#    # time(idx), playerID1, playerID2, ..., playerID 17
#    cols1 = list(team1_players)
#    cols2 = list(team2_players)
#    cols = cols1 + cols2
#    col_num = len(team1_players) + len(team2_players)
#    ddt_dist = pd.DataFrame([[None]*col_num], columns=cols, index=[720.0])
#    print 'number of trajectory data -> ', num
#
#    # read data
#    dist = pkl.load(open('pd_distance.pkl', 'rb'))
#    num = len(dist) - 5
#    # compute the ddt_dist by all the data
#    for i in range(num):
#        print i, '/', num, '\r'
#
#        # get game time
#        timeNow = dist.index[i+4]
#        timeOld = dist.index[i]
#        # exception
#        if timeNow == timeOld:  # if time derivative is impossible
#            continue
#
#        try :
#            dist_tmp = dist.ix[timeOld:timeNow, :]
#        except KeyError:
#            print 'key error occured'
#            print timeNow, timeOld
#
#            # drop None data
#        dist_tmp = dist_tmp.dropna(axis=1)
#
#        # if dist is not 10, skip
#        if dist_tmp.shape[1] != 10:  # if time derivative is impossible
#            continue
#        if dist_tmp.shape[0] != 5:  # if something is wrong...
#            continue
#
#        # compute the derivative
#        tmp = pd.DataFrame([[None]*col_num], columns=cols, index=[timeNow])
#        deltaT = float(timeNow - timeOld)
#        try :
#            # REMARK : needs negative because the time axis is reverted
#            tmp.ix[timeNow, :] = -(dist_tmp.ix[timeNow, :] - dist_tmp.ix[timeOld, :])/ deltaT
#        except KeyError :
#            print 'key error occured'
#            print timeNow, timeOld
#        # update the ddt_dist
#        ddt_dist = ddt_dist.append(tmp)
#
#    # save to pickle
#    pkl.dump(ddt_dist, open('pd_ddt_distance.pkl', 'wb'))



"""
Garbage
"""
#    # simply visualize the ball trajectory
#    
#    # ball data
#    ball = []
#    b_xs = []
#    b_ys = []
#    b_zs = []
#    for i in range(1000): # (num)
#        locs = datatraj[i]['@locations']
#        locs = locs.split(';')
#        print locs[0]
#        b_xs.append(float(locs[0].split(',')[-3]))
#        b_ys.append(float(locs[0].split(',')[-2]))
#        b_zs.append(float(locs[0].split(',')[-1]))
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.scatter(b_xs, b_ys, b_zs, label='Ball', color='Orange')
# #    for i, (x, y, z) in enumerate(zip(c_x, c_y, c_z)):
# #        label = str(i)
# #        ax.text(x, y, z, label)
#     ax.legend()
#     plt.show()
