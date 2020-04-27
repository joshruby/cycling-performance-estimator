# https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku
# https://medium.com/better-programming/keeping-my-heroku-app-alive-b19f3a8c3a82

# Deploying from a subdir of a git repo doesn't work. I followed these tutorials to no avail
# https://www.geekality.net/2019/03/13/heroku-deploy-sub-directory/
# https://medium.com/@shalandy/deploy-git-subdirectory-to-heroku-ea05e95fce1f


import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Methods

# Power output from speed input
# All inputs with units of [%] are entered as decimals, not [%]s
# @st.cache
def p_legs(w, G, v, units='imperial'):
    # Model assumptions
    cda = .509*.63 # = 0.32067
    loss_dt = 0.02 # [2%]
    rho = 1.22601
    c_rr = 0.005

    if(units == 'imperial'):
        w = w / 2.20462
        v = v / 2.23694

    # formula assumes SI units
    output_p = pow((1 - loss_dt), -1) * (9.8067 * w * (math.sin(math.atan(G)) + c_rr * math.cos(math.atan(G))) + 
        (0.5 * cda * rho * pow(v, 2))) * v

    # print('p_legs = ' + str(p_legs))
    return output_p
# test = p_legs(w=150, G=0.081, v=7, units='imperial')
# st.write(test)

# Table of power outputs for incremented speed inputs (constant G)
@st.cache
def p_legs_table(w, G, v_center, units='imperial'):
    speed_window = 2
    speed_increment = 0.25
    input_speeds = np.arange(v_center - speed_window, v_center + speed_window + 1, speed_increment)    
    arr = np.empty((0,3), float)

    for input_v in input_speeds:
      if(input_v > 0):
        output_p = p_legs(w, G, input_v, units)

        if(units == 'metric'):
          vam =  input_v * math.sin(math.atan(G)) * 3600
        elif(units == 'imperial'):
          vam = input_v * 0.44704 * math.sin(math.atan(G)) * 3600

        arr = np.concatenate((arr, [[input_v, output_p, vam]]), axis=0)
        np.set_printoptions(suppress=True)

    return arr
# test = p_legs_table(w=150, G=0.081, v_center=7, units='imperial')
# st.write(test)
###

st.header('Test change')
st.write('Am I there?')
st.writ('How about now?')



data = p_legs_table(w=150, G=0.080, v_center=7, units='imperial')

fig, ax = plt.subplots(constrained_layout=True)
plt.title('Weight of Rider with Gear = 150 lbs, Grade = 8.0%')

ax.plot(data[:, 1], data[:, 0])
ax.set_xlabel("Power", fontsize=14)
ax.set_ylabel("Speed", fontsize=14)

# These methods are only used to set up the double-x axis in the plot
# Kind of annoying that this is how you do it
def power_to_powerperkg(x):
    return x / (150 / 2.20462) # if units = imperial
    return x / 150 # if units = metric

def powerperkg_to_power(x):
    return x * (150 / 2.20462) # if units = imperial
    return x * 150 # if units = metric

# Second x axis that lives above the top line
secax = ax.secondary_xaxis('top', functions=(power_to_powerperkg, powerperkg_to_power))
secax.set_xlabel('Power/kg')

# Second y axis that lives on the right of plot
ax2=ax.twinx()
ax2.plot(data[:, 1], data[:, 2])
ax2.set_ylabel("VAM", fontsize=14)

plt.grid(True)
# plt.show()

st.write(fig)

st.subheader('Time to finish OLH (Strava ID 8109834)')
data = p_legs_table(150, 0.08, 7, units = 'imperial')
ttf = 2.98 / data[:, 0] * 60
data = np.column_stack((data, ttf))
data = pd.DataFrame(data,
                    columns = ['Speed [m/s]', 'Power [W]', 'VAM [m/hr]', 'Time [min]']
)
st.table(data)