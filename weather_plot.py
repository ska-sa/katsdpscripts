#!/usr/bin/env python
import time
import numpy as np
import matplotlib
import random
import datetime
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

import ffuilib as ffui

enviro = ffui.build_device("enviro","ff-proxy",1258)
 # connect to antenna 1 to get environment data
enviro.sensor_enviro_wind_speed.set_strategy("period","1000")
enviro.sensor_enviro_wind_direction.set_strategy("period","1000")

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax.set_title("Wind Data")
ax.set_ylabel("Speed (m/s)", color='b')
ax2 = ax.twinx()
ax2.set_ylabel("Direction (deg)", color='r')

def get_wind():
    return (enviro.sensor_enviro_wind_speed.value, enviro.sensor_enviro_wind_direction.value)

def format_date(x, pos=None):
    return matplotlib.dates.num2date(x).strftime('%H:%M:%S')

def animate():
    tstart = time.time()
    initial_x = np.arange(tstart-60,tstart).tolist()
    tstamp_x = [datetime.datetime.fromtimestamp(q) for q in initial_x]
    ws = np.zeros(60).tolist()
    wd = np.zeros(60).tolist()
    line, = ax.plot(tstamp_x, ws, 'b')
    line2, = ax2.plot(tstamp_x, wd, 'r')
    ax.set_ylim(ymin=0, ymax=30)
    ax2.set_ylim(ymin=0, ymax=360)

    for i in np.arange(1,200):
        wind = get_wind()
        ws.pop(0)
        ws.append(wind[0])
        wd.pop(0)
        wd.append(wind[1])
        initial_x.pop(0)
        initial_x.append(time.time())
        tstamp_x = [datetime.datetime.fromtimestamp(q) for q in initial_x]
        line.set_data(tstamp_x, ws)
        line2.set_data(tstamp_x, wd)
        ax.set_xlim(xmin=tstamp_x[0], xmax=tstamp_x[-1])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        fig.canvas.draw()
        #fig.canvas.blit(ax.bbox)
        time.sleep(1)

win = fig.canvas.manager.window
fig.autofmt_xdate()
fig.canvas.manager.window.after(100, animate)
plt.show()
