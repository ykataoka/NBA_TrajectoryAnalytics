#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xmltodict
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
import cPickle as pkl


def plot_XYtrajectory(Xs, Ys, starttime, endtime):
    """
    @desc : plot trajectory data given the start and end time
    @param Xs : x coordinate of ball and players
    @param Ys : y coordinate of ball and players
    @param startime : starttime plot
    @param endtime : starttime plot
    """
    # Xs, Ys
    X = Xs.ix[starttime:endtime, :]
    Y = Ys.ix[starttime:endtime, :]

    # visualize
    title = 'trajectory of ball and players'
    plt.title(title)
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')

    # revert x axis
#    X = X.dropna()
#    Y = Y.dropna()
    X = X.dropna(axis=1)
    Y = Y.dropna(axis=1)
    pkl.dump(X, open('pd_1sec_X_test.pkl', 'wb'))
    pkl.dump(Y, open('pd_1sec_Y_test.pkl', 'wb'))

    plt.plot(X, Y)
    plt.grid(True)
#    plt.legend()
#    plt.ylim(0, 5)
    plt.show()


def dump_XYdata2pkl():
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
    compute distance between ball and players (dist)
    """
    # set the pandas data structure to save distance ball (dist)
    # Xs : time(idx), ballID_x, playerID1_x, ..., playerID17_x
    # Ys : time(idx), ballID_x, playerID1_y, ..., playerID17_y
    cols1 = list(team1_players)
    cols2 = list(team2_players)
    cols = ['ball'] + cols1 + cols2
    col_num = 1 + len(team1_players) + len(team2_players)
    Xs = pd.DataFrame([[None]*col_num], columns=cols, index=[720.0])
    Ys = pd.DataFrame([[None]*col_num], columns=cols, index=[720.0])

    # insert data one by one
    for i in range(num):
        print i, '/', num, '\r'
        # get game time
        timeNow = float(datatraj[i]['@game-clock'])

        # get locations
        locs = datatraj[i]['@locations']
        locs = locs.split(';')
        ballX, ballY = 0, 0
        tmp_X = pd.DataFrame([[None]*col_num], columns=cols, index=[timeNow])
        tmp_Y = pd.DataFrame([[None]*col_num], columns=cols, index=[timeNow])
        for loc in locs:  # loop 11 datas
            loc_list = loc.split(',')  # [23,262882,20.47625,21.10303,0]
            if loc_list[0] == '-1':  # ball
                ballX = float(loc_list[2])
                ballY = float(loc_list[3])
                tmp_X.ix[timeNow, 'ball'] = ballX
                tmp_Y.ix[timeNow, 'ball'] = ballY
            else:  # player's trajectory
                playerID = loc_list[1]
                playerX = float(loc_list[2])
                playerY = float(loc_list[3])
                tmp_X.ix[timeNow, playerID] = playerX
                tmp_Y.ix[timeNow, playerID] = playerY

        # update the dist
        Xs = Xs.append(tmp_X)
        Ys = Ys.append(tmp_Y)

    # dump data
    pkl.dump(Xs, open('pd_Xs.pkl', 'wb'))
    pkl.dump(Ys, open('pd_Ys.pkl', 'wb'))


if __name__ == '__main__':

    """
    read from pickle
    """
    Xs = pkl.load(open('pd_Xs.pkl', 'rb'))
    Ys = pkl.load(open('pd_Ys.pkl', 'rb'))

    """
    read from original trajectory data
    """
#    dump_XYdata2pkl()

    # plot trajectory
    plot_XYtrajectory(Xs, Ys, 610.00, 609.00)
