''' Create a simple stress analytics visualization on webUI.

    To execute this file, do the following command on this directory
    $ cd rep1
    $ bokeh serve --show main.py

    To access the web page
    http://localhost:5006/main2
'''
try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print ("WARNING: Cache for this example is available on Python 3 only.")
    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

from os.path import dirname, join
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc


from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import PreText, Select
from bokeh.models import ColumnDataSource, HoverTool, Div, Range1d, PanTool, BoxZoomTool, ResizeTool, ResetTool, WheelZoomTool
from bokeh.plotting import figure
from bokeh.models import Range1d
import cPickle as pkl

# configuration
pd.options.display.float_format = '{:.2f}'.format

# parameter
DATA_DIR = join(dirname(__file__), 'data_viz')
DEFAULT_TICKERS_1 = ['userK', 'userB', 'userI']
DEFAULT_TICKERS_2 = ['driver', 'frontPsg', 'rearPsg']
DEFAULT_TICKERS_3 = ['normal', 'oneStop', 'crazy']
DUMMY_TICKERS_0  = ['LooseBall_id_1',
                    'LooseBall_id_2',
                    'LooseBall_id_3',
                    'LooseBall_id_4',
                    'LooseBall_id_5',
                    'LooseBall_id_6',
                    'LooseBall_id_7',
                    'LooseBall_id_8']
DUMMY_TICKERS_1 = ['20160126', '20160202', '20160205', '20160206', '20160211']
DUMMY_TICKERS_2 = ['CHA_vs_SAC',
                   'CHA_vs_LAL',
                   'CHI_vs_SAC',
                   'NO_vs_CLE',
                   'HOU_vs_POR']
DUMMY_TICKERS_3 = ['1', '2', '3', '4', '5', '6']

time_init_default = 0
normal_constant = pd.read_csv('data_viz/normal_constant.csv')
normal_constant = normal_constant.set_index('Unnamed: 0')
normal_constant = normal_constant.rename_axis("role")


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


@lru_cache()
def load_ticker(t1, t2, t3):
    fname = join(DATA_DIR, '%s_%s_%s.csv' % (t1, t2, t3) )
    data  = pd.read_csv(fname)
    time_init = data['Time'][0] # time index
    time_corr = (data['Time'] - time_init) / 1000.0
    time_init_default = time_init
    data['Time'] = time_corr
    return data


@lru_cache()
def get_data(t1, t2, t3):
    df = load_ticker(t1, t2, t3)
    data = df     # preprocess data, if necessary
    return data


# set up widgets
looseBall_title_s = PreText(text='Find Loose Ball ',
                            width=1100, height=10)
trajectory_title_s = PreText(text='Loose Ball Trajectory',
                             width=1100, height=10)
distance_title_s = PreText(text='Distance between Ball and Players',
                           width=1100, height=10)
direction_title_s = PreText(text='Direction Difference between Ball and Players',
                            width=1100, height=10)
derivative_title_s = PreText(text='Velocity between Ball and Players(Derivative of Distance)',
                             width=1100, height=10)

stats_title_s = PreText(text='Selected Data of current setting', width=600, height=5)
stats_title_a = PreText(text='All Data of current setting',    width=600, height=5)
stats_title_n = PreText(text="Default Stress Level", width=600, height=5)
stats_select  = PreText(text='', width=600)
stats_all     = PreText(text='', width=600)
stats_normal  = PreText(text='', width=600)

# original ticker (work)
ticker1 = Select(title="user", value='userB', options=DEFAULT_TICKERS_1, width=100)
ticker2 = Select(title="role", value='driver', options=DEFAULT_TICKERS_2, width=100)
ticker3 = Select(title="expiment type", value='crazy', options=DEFAULT_TICKERS_3, width=100)

# dummy ticker
ticker_d0 = Select(title="Loose Ball", value='LooseBall_id_1',
                   options=DUMMY_TICKERS_0, width=260)
ticker_d1 = Select(title="Game Date", value='20160125',
                   options=DUMMY_TICKERS_1, width=260)
ticker_d2 = Select(title="Home vs Visit", value='GSW_vs_LAL',
                   options=DUMMY_TICKERS_2, width=260)
ticker_d3 = Select(title="Quarter", value='4',
                   options=DUMMY_TICKERS_3, width=260)

# set up data
source    = ColumnDataSource(data=dict(Time=[], ecg=[], acc_all=[], hr=[], rri=[], stress=[]))
source_static = ColumnDataSource(data=dict(Time=[], ecg=[], acc_all=[], hr=[], rri=[], stress=[]))
ecg       = ColumnDataSource(data=dict(Time=[], ecg=[]))
acc       = ColumnDataSource(data=dict(Time=[], acc_all=[]))
hr        = ColumnDataSource(data=dict(Time=[], hr=[]))
rri       = ColumnDataSource(data=dict(Time=[], rri=[]))
srs       = ColumnDataSource(data=dict(Time=[], stress=[]))
srs_out   = ColumnDataSource(data=dict(Time=[], stress=[]))
car       = ColumnDataSource(data=dict(Time=[], Longitude=[], Latitude=[],
                                       GPSSpeed=[], 
                                       Altitude=[], Bearing=[], Gx=[], Gy=[], Gz=[],
                                       Gcalibrated=[], AccelerationSensor=[],
                                       EngineRPM=[], GPSAccuracy=[], GPSAltitude=[],
                                       GPSBearing=[], GPSLatitude=[], GPSLongitude=[], Speed=[] ))
car_gps   = ColumnDataSource(data=dict(Time=[], Longitude=[], Latitude=[]))
scale_ecg = ColumnDataSource(data=dict(Time=[], scale_ecg=[]))
scale_acc = ColumnDataSource(data=dict(Time=[], scale_acc=[]))
scale_hr  = ColumnDataSource(data=dict(Time=[], scale_hr=[]))
scale_rri = ColumnDataSource(data=dict(Time=[], scale_rri=[]))
scale_srs = ColumnDataSource(data=dict(Time=[], scale_srs=[]))


print "initialize the data frame"
ecg_df = pd.DataFrame([])
acc_df = pd.DataFrame([])
hr_df  = pd.DataFrame([])
rri_df = pd.DataFrame([])
srs_df = pd.DataFrame([])


# set up tools
tools  = 'pan,wheel_zoom,box_zoom,xbox_select,reset'
tools2 = 'pan,box_zoom,xbox_select,reset'
tools3 = 'pan,wheel_zoom,box_zoom,box_select,reset'

def ticker1_change(attrname, old, new):
    update()

def ticker2_change(attrname, old, new):
    update()

def ticker3_change(attrname, old, new):
    update()

def std_normalize(A):
    return ( A - np.average(A) ) / np.std(A)
    
def update(selected=None):
    print 'update is called.'
    
    # read data
    t1, t2, t3 = ticker1.value, ticker2.value, ticker3.value
    data = get_data(t1, t2, t3)
    
    # global 
    global ecg_df
    global acc_df
    global hr_df 
    global rri_df
    global srs_df
    global s_ecg_df 
    global s_acc_df 
    global s_hr_df 
    global s_rri_df 
    global s_srs_df 


    # update the default stress level
    constant     = normal_constant.ix[t2,t1] # this value is obtained from only normal condition
    constant     = [float(x) for x in constant[1:-1].split(',')]
    constant_all = normal_constant.ix['all',t1]
    constant_all = [float(x) for x in constant_all[1:-1].split(',')]
    c = pd.DataFrame([constant_all, constant], columns=['count', 'mean', 'std', 'max', 'min'], index=['All','OnlyNormal'])
    update_stats( c, 'normal' )

    # update the threshold
    alpha        = constant[1] + 1.*constant[2] # reliablity range 68%
    
    # drop None
    ecg_df = data[['Time', 'ecg']].dropna().set_index('Time')
    acc_df = data[['Time', 'acc_all']].dropna().set_index('Time')
    hr_df  = data[['Time', 'hr' ]].dropna().set_index('Time')
    rri_df = data[['Time', 'rri']].dropna().set_index('Time')
    srs_df = data[['Time', 'stress']].dropna().set_index('Time')
    srs_df['thres'] = alpha
    srs_out_df = srs_df[srs_df['stress'] > alpha]

    # scale
    s_ecg_df = std_normalize(ecg_df)
    s_acc_df = std_normalize(acc_df)
    s_hr_df  = std_normalize(hr_df)
    s_rri_df = std_normalize(rri_df)
    s_srs_df = std_normalize(srs_df)

    # change to ColumnDataSource
    ecg.data = ecg.from_df(ecg_df)
    acc.data = acc.from_df(acc_df)
    hr.data  =  hr.from_df(hr_df)
    rri.data = rri.from_df(rri_df)
    srs.data = srs.from_df(srs_df)
    srs_out.data = srs_out.from_df(srs_out_df)

    car_gps.data = car_gps.from_df(data[['Time', 'Latitude', 'Longitude']].dropna()) 
    time_texts = [int(float(x)) for x in car_gps.data['Time']]
    car_gps.data['Time'] = time_texts
    
    # compute the scaled values
    scale_ecg.add(ecg.data['Time'], name='Time')
    scale_ecg.add(s_ecg_df['ecg'], name='scale_ecg')
    scale_acc.add(acc.data['Time'], name='Time')
    scale_acc.add(std_normalize(acc.data['acc_all']), name='scale_acc')
    scale_hr.add(hr.data['Time'], name='Time')
    scale_hr.add(std_normalize(hr.data['hr']), name='scale_hr')
    scale_rri.add(rri.data['Time'], name='Time')
    scale_rri.add(std_normalize(rri.data['rri']), name='scale_rri')
    scale_srs.add(srs.data['Time'], name='Time')
    scale_srs.add(std_normalize(srs.data['stress']), name='scale_srs')
    
    #stats indication
    stats_data = pd.concat([ecg_df['ecg'], acc_df['acc_all'], hr_df['hr'], rri_df['rri'], srs_df['stress']], axis=1)
    update_stats( stats_data, 'all' )
    
    
def update_stats(data, flg):
    print 'update stats is called'
#    print data['ecg'].describe().transpose()
#    print data['acc_all'].describe()
#    print ecg_df.describe()
#    print scale_srs.data['ecg'].describe() # describe is the method of pd.DataFrame
    if flg == 'all':
        stats_all.text = str(data.describe().transpose()[['count','mean','std','max','min']])
    elif flg == 'select':
        stats_select.text = str(data.describe().transpose()[['count','mean','std','max','min']])
    elif flg == 'normal':
        stats_normal.text = str(data)

def selection_change(attrname, old, new):
    print 'selection is changed!'
    t1, t2, t3 = ticker1.value, ticker2.value, ticker3.value
    data = get_data(t1, t2, t3)

    # value
    tmp_srs_df = srs_df
    tmp_ecg_df = ecg_df
    tmp_acc_df = acc_df
    tmp_rri_df = rri_df
    tmp_hr_df  = hr_df

    # update values individually
    if srs.selected['1d']['indices'] != []:
        selected = srs.selected['1d']['indices']
        min_selected = min(selected)
        max_selected = max(selected)
        t_min_selected = srs.data['Time'][min_selected]
        t_max_selected = srs.data['Time'][max_selected]
        tmp_srs_df = srs_df.ix[t_min_selected:t_max_selected, :]
        
    if hr.selected['1d']['indices'] != []:
        selected = hr.selected['1d']['indices']
        min_selected = min(selected)
        max_selected = max(selected)
        t_min_selected = hr.data['Time'][min_selected]
        t_max_selected = hr.data['Time'][max_selected]
        tmp_hr_df  = hr_df.ix[t_min_selected :t_max_selected, :]

    if ecg.selected['1d']['indices'] != []:
        selected = ecg.selected['1d']['indices']
        min_selected = min(selected)
        max_selected = max(selected)
        t_min_selected = ecg.data['Time'][min_selected]
        t_max_selected = ecg.data['Time'][max_selected]
        tmp_ecg_df  = ecg_df.ix[t_min_selected :t_max_selected, :]

    if acc.selected['1d']['indices'] != []:
        selected = acc.selected['1d']['indices']
        min_selected = min(selected)
        max_selected = max(selected)
        t_min_selected = acc.data['Time'][min_selected]
        t_max_selected = acc.data['Time'][max_selected]
        tmp_acc_df  = acc_df.ix[t_min_selected :t_max_selected, :]

    if rri.selected['1d']['indices'] != []:
        selected = rri.selected['1d']['indices']
        min_selected = min(selected)
        max_selected = max(selected)
        t_min_selected = rri.data['Time'][min_selected]
        t_max_selected = rri.data['Time'][max_selected]
        tmp_rri_df  = rri_df.ix[t_min_selected :t_max_selected, :]        
        
    #stats indication
    stats_data_selected = pd.concat([tmp_ecg_df['ecg'], tmp_acc_df['acc_all'], tmp_hr_df['hr'], tmp_rri_df['rri'], tmp_srs_df['stress']], axis=1)
    update_stats(stats_data_selected, 'select')
    
### main ###

# configuration on change
ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)
ticker3.on_change('value', ticker3_change)
srs.on_change('selected', selection_change)
hr.on_change('selected', selection_change)
ecg.on_change('selected', selection_change)
acc.on_change('selected', selection_change)
rri.on_change('selected', selection_change)

scale_srs.on_change('selected', selection_change)

# initialization
update()


"""
Parameter setting
"""

# GOOD, rebound.
#starttime = 550.69
#endtime = 549.69

# GOOD : 4 people try to take it
starttime = 398.25
endtime = 397.24

# GOOD, rebound
#starttime = 34.60
#endtime = 33.60


"""
Distance Graph
"""
# read distance data
dst = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_distance.pkl', 'rb'))
print dst
dst = dst.ix[starttime:endtime, :]
dst = dst.dropna(axis=1)

# set distance graph
fig_dst = figure(plot_width=500,
                 plot_height=300,
                 tools=tools,
                 active_drag="box_zoom")
fig_dst.logo = None
fig_dst.xaxis.axis_label = 'Remaining Game Time [sec]'
fig_dst.yaxis.axis_label = 'Distance [ft]'
fig_dst.xaxis.axis_label_text_font_size = "14pt"
fig_dst.yaxis.axis_label_text_font_size = "14pt"


# plot distance
color_teamA = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']  # blue
color_teamB = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476']  # green
colors_dst = color_teamA + color_teamB
for i in range(len(colors_dst)):
    fig_dst.line(np.array(dst.index),
                 np.array(dst.ix[:, i]),
#                 legend=legends_dst[i],
                 color=colors_dst[i],
                 line_width=2)

fig_dst.set(x_range=Range1d(starttime, endtime))
#fig_dst.set(y_range=Range1d(0, 50))

# set threshold
threshold = 10  # [ft]

fig_dst.patches([[min(dst.index), max(dst.index),
                  max(dst.index), min(dst.index)]],
                [[0, 0, threshold, threshold]],
                color=["firebrick"], alpha=[0.3], line_width=2)

"""
Derivative of Distance(Velocity) Graph
"""
# read derivative of distance data
ddt_dst = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_ddt_distance.pkl', 'rb'))
ddt_dst = ddt_dst.ix[starttime:endtime, :]
ddt_dst = ddt_dst.dropna(axis=1)

# set derivative of distance graph
fig_ddt_dst = figure(plot_width=500,
                     plot_height=300,
                     tools=tools,
                     active_drag="box_zoom")
fig_ddt_dst.logo = None
fig_ddt_dst.xaxis.axis_label = 'Remaining Game Time [sec]'
fig_ddt_dst.yaxis.axis_label = 'Velocity [ft/sec]'
fig_ddt_dst.xaxis.axis_label_text_font_size = "14pt"
fig_ddt_dst.yaxis.axis_label_text_font_size = "14pt"


# plot derivative of distance
color_teamA = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']  # blue
color_teamB = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476']  # green
colors_ddt_dst = color_teamA + color_teamB
for i in range(len(colors_ddt_dst)):
    fig_ddt_dst.line(np.array(ddt_dst.index),
                     np.array(ddt_dst.ix[:, i]),
                     #                 legend=legends_ddt_dst[i],
                     color=colors_ddt_dst[i],
                     line_width=2)

fig_ddt_dst.set(x_range=Range1d(starttime, endtime))
fig_ddt_dst.set(y_range=Range1d(ddt_dst.min().min(), ddt_dst.max().max()))

# set threshold
threshold = -10  # [ft/sec]
fig_ddt_dst.patches([[min(ddt_dst.index), max(ddt_dst.index),
                      max(ddt_dst.index), min(ddt_dst.index)]],
                    [[-100, -100,
                      threshold, threshold]],
                    color=["firebrick"], alpha=[0.3], line_width=2)

# legend location, delete logo
# fig_dst.legend.orientation = "horizontal"
# fig_dst.legend.location = "bottom_center"


"""
Direction Graph
"""
# read direction data
dirc = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_direction.pkl', 'rb'))
dirc = dirc.ix[starttime:endtime, :]
dirc = dirc.dropna(axis=1)

# set direction graph
fig_dirc = figure(plot_width=500,
                  plot_height=300,
                  tools=tools,
                  active_drag="box_zoom")
fig_dirc.logo = None
fig_dirc.xaxis.axis_label = 'Remaining Game Time [sec]'
fig_dirc.yaxis.axis_label = 'Direction Difference [deg]'
fig_dirc.xaxis.axis_label_text_font_size = "14pt"
fig_dirc.yaxis.axis_label_text_font_size = "14pt"

# plot direction
color_teamA = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']  # blue
color_teamB = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476']  # green
colors_dirc = color_teamA + color_teamB
for i in range(len(colors_dirc)):
    fig_dirc.line(np.array(dirc.index),
                  np.array(dirc.ix[:, i]),
#                 legend=legends_dirc[i],
                  color=colors_dirc[i],
                  line_width=2)

# add threshold line and patch
threshold = 20

fig_dirc.patches([[min(dirc.index), max(dirc.index),
                   max(dirc.index), min(dirc.index)]],
                 [[0, 0, threshold, threshold]],
                 color=["firebrick"], alpha=[0.3], line_width=2)

fig_dirc.set(x_range=Range1d(starttime, endtime))
#fig_dirc.set(y_range=Range1d(0, 50))

"""
Plot Trajectory & Court
"""
hover = HoverTool(tooltips=[
    ("Player", "@player"),
    ("X", "@x"),
    ("Y", "@y"),
])

# define the figure
p_court = figure(plot_height=850,
                 plot_width=1040,
                 toolbar_sticky=False,
                 tools=tools,
                 active_drag="box_zoom")
p_court.xaxis.axis_label = 'x [ft]'
p_court.yaxis.axis_label = 'y [ft]'
p_court.xaxis.axis_label_text_font_size = "14pt"
p_court.yaxis.axis_label_text_font_size = "14pt"


# Court
p_court.line(x=[0, 94], y=[0, 0], line_width=3, line_color='gray')
p_court.line(x=[94, 94], y=[0, 50], line_width=3, line_color='gray')
p_court.line(x=[0, 94], y=[50, 50], line_width=3, line_color='gray')
p_court.line(x=[0, 0], y=[0, 50], line_width=3, line_color='gray')
p_court.line(x=[47, 47], y=[0, 50], line_width=3, line_color='gray')
p_court.circle(x=47, y=25, size=100, color=None, line_color='gray',
               line_width=3, fill_alpha=0.5)  # center circle

# Goal 1
p_court.circle(x=88, y=25, size=30, color="gray", line_color='orange',
               line_width=5, fill_alpha=0.50)
p_court.line(x=[90, 90], y=[20, 30], color="gray", line_width=8)
# Goal 2
p_court.circle(x=6, y=25, size=30, color="gray", line_color='orange',
               line_width=5, fill_alpha=0.50)
p_court.line(x=[4, 4], y=[20, 30], color="gray", line_width=8)

# read trajectory of ball and players on court
X_org = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_Xs.pkl', 'rb'))
Y_org = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_Ys.pkl', 'rb'))
Xs = X_org.ix[starttime:endtime, :]
Ys = Y_org.ix[starttime:endtime, :]
Xs = Xs.dropna(axis=1)
Ys = Ys.dropna(axis=1)

# colors
Ball = ['#ef6548']  # orange
color_teamA = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']  # blue
color_teamB = ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476']  # green

# legends
legends = ['Ball', 'A_p1', 'A_p2', 'A_p3', 'A_p4', 'A_p5',
           'B_p1', 'B_p2', 'B_p3', 'B_p4', 'B_p5']

# plot ball and players respectively
for i in range(11):
    # X, Y
    X = Xs.iloc[:, i]
    Y = Ys.iloc[:, i]

    # color
    if i == 0:
        colornow = Ball[0]
    elif 1 <= i and i <= 5:
        colornow = color_teamA[i-1]
    elif 6 <= i and i <= 10:
        colornow = color_teamB[i-6]

    # size
    sizes = [20 * j / float(len(X)) for j in range(len(X))]

    # plot circles for trajectory
    if i == 0:
        p_court.circle(np.array(X), np.array(Y), size=np.array(sizes)*1.5, color=colornow,
                       line_color='black', line_width=3, alpha=0.5, legend=legends[i])
    else:
        p_court.circle(np.array(X), np.array(Y), size=sizes, color=colornow,
                       line_color='gray', line_width=1, alpha=0.5, legend=legends[i])

# legend location, delete logo
p_court.legend.orientation = "horizontal"
p_court.legend.location = "bottom_center"
p_court.logo = None

# Fix the width and height
left, right, bottom, top = -5, 99, -10, 55
p_court.set(x_range=Range1d(left, right), y_range=Range1d(bottom, top))


"""
Set up overall layout
"""
param_title = looseBall_title_s
trajectory_title = trajectory_title_s
params = row(ticker_d1, ticker_d2, ticker_d3, ticker_d0)
widgets = column(param_title,
                 params,
                 trajectory_title,
                 p_court)
# stats_title_s, stats_select, stats_title_n, stats_normal

series = column(direction_title_s,
                fig_dirc,
                distance_title_s,
                fig_dst,
                derivative_title_s,
                fig_ddt_dst)
layout = row(widgets, series)

# rendering
curdoc().add_root(layout)
curdoc().title = "Measure Loose Ball! SSAC'17 Hackathon"
