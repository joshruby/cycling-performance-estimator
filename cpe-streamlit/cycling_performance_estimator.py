# v1.0
version = 'v1.0'

#############################################################################################################
# https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku
# https://medium.com/better-programming/keeping-my-heroku-app-alive-b19f3a8c3a82

# Deploying from a subdir of a git repo doesn't work. I followed these tutorials to no avail
# https://www.geekality.net/2019/03/13/heroku-deploy-sub-directory/
# https://medium.com/@shalandy/deploy-git-subdirectory-to-heroku-ea05e95fce1f

# To manually update the Heroku deployment:
# $ heroku login
# $ heroku git:clone -a agile-journey-39962
# $ cd agile-journey-39962

# Now manually delete the contents of the folder that was created and copy in the latest files in the working dir of the app

# $ git add .
# $ git commit -am "make it better"
# $ git push heroku master
#############################################################################################################


#############################################################################
# Oauth2
# http://developers.strava.com/docs/authentication/
# https://gist.github.com/frankie567/63d499a288e2858869c062b2c652d0fd

import streamlit as st
import copy
import math
import numpy as np
# from sympy.solvers import solveset
# from sympy import Symbol
# import matplotlib.pyplot as plt
import pandas as pd
import requests
# from requests_oauthlib import OAuth2Session
import shelve
import polyline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from haversine import haversine, Unit
# from scipy.interpolate import interp1d
import os
import json
import time

mapbox_token = 'pk.eyJ1IjoianJydWJ5IiwiYSI6ImNrOWtrMDU3czF2dTkzZG53Nmw2NDdneTMifQ.zzXEhr0Z1biR2pydOFco8A'

### Load segments into one text file
# segmentIDs = [620770, 609300, 6445920, 1012036, 627158, 642780, 614445, 651158, 726454, 628189, 3615418, 7684550, 794318, 646188, 627058, 3621774, 617901, 226705, 359323, 4235382, 617794, 2149043, 616252, 611413, 1638547, 8109834, 4197]

# # Make dict
# keys = []
# values = []
# for segmentID in segmentIDs:
#     headers = {"Authorization": "Bearer 7f0df2a26f507385948517b86ac51b29682be4b4"}
#     call = 'https://www.strava.com/api/v3/segments/'+str(segmentID)
#     r = requests.get(call, headers=headers)
    
#     r = r.json()
#     values.append(r)

#     segmentName = r['name']
#     keys.append(segmentName)
# segmentDict = dict(zip(keys, values))

# # Write to text file in json format
# with open('segmentDict.txt', 'w') as outfile:
#     json.dump(segmentDict, outfile)

# # Make shelve
# with shelve.open('test') as db:
#     db[segmentName] = r
#     # st.subheader('Shelve value')
#     # st.write(db[segmentName])
#     # del db['Skyline - Old La Honda to Page Mill']
#     st.subheader('segments.db keys')
#     st.write(list(db.keys()))
#     st.write(db['Test'])

###

### Global variables
# unit = st.sidebar.radio('Units', ['Metric', 'Imperial'])
w_person = st.sidebar.number_input('Rider Weight [kg]', value=0.)
w_gear = st.sidebar.number_input('Gear Weight [kg]', value=11.)
w_total = w_person + w_gear

cda = st.sidebar.number_input('Cd A [m^2]', value=0.321) # .509*.63 = 0.32067
rho = st.sidebar.number_input('Air Density [kg/m^3]', value=1.226)
loss_dt = st.sidebar.number_input('Drivetrain Loss [%]', value=5) / 100
crr = st.sidebar.number_input('Coefficient of Rolling Resistance [%]', value=0.5) / 100

st.sidebar.markdown('*{}*'.format(version))
st.sidebar.markdown('*For help or to report a bug, email me at josh.r.ruby@gmail.com.*')

# # From this point forward all quanitites are assumed to be metric by default. If imperial is selected about the model outputs will need to be converted before being displayed.  
# if(unit == 'imperial'):
#     w = w * 0.454
# ###

### Methods
def p_model( loss_dt, w, G, crr, cda, rho, v ):
    return pow((1 - loss_dt), -1) * (9.8067 * w * (math.sin(math.atan(G)) + crr * math.cos(math.atan(G))) + 
            (0.5 * cda * rho * pow(v, 2))) * v

# Power output from speed input
# Assumes SI unit inputs (kg, m/s)
# All inputs with units of [%] are entered as decimals, not [%]
# cda, loss_dt, rho, and crr are global variables defined outside the func 
def p_legs( w, G, v):
    p = p_model(loss_dt, w, G, crr, cda, rho, v)
    return p

# Table of power outputs for incremented speed inputs (constant G)
# Assumes SI unit inputs [m/s]
@st.cache
def p_legs_table( w, G ):
    df = pd.DataFrame()

    v_min = 0
    v_max = 100
    increment = 0.01
    input_speeds = np.arange(v_min, v_max + increment, increment)
    
    for input_v in input_speeds:
        v_kmh = input_v * 3.6
        v_mph = input_v * 2.237
        output_p = p_legs(w, G, input_v)
        vam = input_v * math.sin(math.atan(G)) * 3600

        df_newRow = pd.DataFrame([[input_v, v_kmh, v_mph, output_p, vam]], columns=['v [m/s]', 'v [km/h]', 'v [mi/h]', 'p [W]', 'vam [m/h]'])
        df = df.append(df_newRow, ignore_index=True)
        
        # Break the function when output_p reaches 500 W
        if (df_newRow['p [W]'] > 500).bool():
            return df
    return df

# @st.cache
def speedVsPowerPlot( w, G ):
    df = p_legs_table(w=w, G=G)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['p [W]'],
        y=df['v [km/h]'],
        name='km/h',
        # hovertemplate= '%{kmh} km/h <br> %{vam} VAM <extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['p [W]'],
        y=df['v [mi/h]'],
        name='mi/h',
        # fill='tozeroy',
    ))
    fig.update_layout(
        title='Speed vs Power',
        xaxis_title='Power [W]',
        yaxis_title='Speed [km/h]',
        hovermode='x',
        # plot_bgcolor='white'
    )

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # data = go.Scatter()
    # fig.add_trace(go.Scatter(
    #     x=df['p [W]'],
    #     y=df['vam [m/h]'],
    #     name='VAM',
    #     showlegend=False,
    #     line=dict(width=0.001)
    #     # visible=False
    # ), secondary_y=True)
    # fig.add_trace(go.Scatter(
    #     x=df['p [W]'],
    #     y=df['v [km/h]'],
    #     name='km/h',
    # ), secondary_y=False)
    # # fig.add_trace(go.Scatter(
    # #     x=df['p [W]'],
    # #     y=df['v [mi/h]'],
    # #     name='mi/h',
    # # ), secondary_y=False)
    # fig.update_layout(
    #     title='Total Weight: {} kg  <br>  Grade: {}%'.format(str(w), str(G*100)),
    #     xaxis_title='Power [W]',
    #     hovermode='x',
    #     # plot_bgcolor='white'
    # )
    # fig.update_yaxes(title_text='Speed [km/h]', secondary_y=False)
    # fig.update_yaxes(title_text='VAM [m/h]', showgrid=False, secondary_y=True)


    # # https://plotly.com/python/hover-text-and-formatting/
    # fig = go.Figure()
    # temp = df['vam [m/h]']
    # fig.add_trace(go.Scatter(
    #     x=df['p [W]'],
    #     y=df['v [km/h]'],
    #     name='km/h',
    #     customdata={[df['vam [m/h]'], df['vam [m/h]']]},
    #     hovertemplate='Something <br>%{customdata[0]:.2f} VAM<extra></extra>'       
    # ))
    # fig.add_trace(go.Scatter(
    #     x=df['p [W]'],
    #     y=df['v [mi/h]'],
    #     name='mi/h',
    # ))
    # fig.update_layout(
    #     title='Total Weight = {} kg, Grade = {}%'.format(str(w), str(G*100)),
    #     xaxis_title='Power [W]',
    #     yaxis_title='Speed [km/h]',
    #     hovermode='x',
    #     # plot_bgcolor='white'
    # )
   
    return fig

def vamPlot( w, G ):
    df = p_legs_table(w=w, G=G)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['p [W]'],
        y=df['vam [m/h]'],
        name='vam',
    ))
    fig.update_layout(
        title='VAM vs Power',
        xaxis_title='Power [W]',
        yaxis_title='VAM [m/h]',
        hovermode='x',
        # plot_bgcolor='white'
    )
    
    return fig

# Assumes s units (d [m])
# Returns p_legs_table array with an additional column called 'timeToFinish' with units of [s]
@st.cache
def timeToFinish( w, G, d ):
    df = copy.deepcopy(p_legs_table(w=w, G=G))
    df['timeToFinish [min]'] = (d / df['v [m/s]']) / 60
    df = df.loc[df['p [W]'] >= 80]
    return df

# @st.cache
def timeToFinishPlot( w, G, d ):
    df = timeToFinish(w=w, G=G, d=d)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['p [W]'],
        y=df['timeToFinish [min]'],
        # name=str(w-i) + ' kg',
        # fill='tonexty',
    ))
    fig.update_layout(
        title='Time to Finish vs Power',
        xaxis_title='Power [W]',
        yaxis_title='Time to Finish [min]',
        hovermode='x',
        # plot_bgcolor='white'
    )
    return fig

@st.cache
def segmentMap( db, key ):
    # returns a list of decoded (lat, lon) tuples
    segment_path = polyline.decode(db[key]['map']['polyline'])
    segment_path = np.array(segment_path)
    segment_haversine = haversine(db[key]['start_latlng'], db[key]['end_latlng'], unit=Unit.METERS)
    # Fitting haversine into 400 px
    segment_haversine_per_px = segment_haversine/400
    # st.write('Meters/pixel needed to fit haversine in 400 pixels = haversine/400 = ', segment_haversine_per_px)
    # Programmatically determine the optimal zoom level (assuming 40 deg latitude)
    # See here: https://docs.mapbox.com/help/glossary/zoom-level/
    # 29.277 corresponds to a zoom level of 10 at 40 deg lat
    starting_span = 29.277
    for zoom_level in range(10, 18+1):
        i = zoom_level - 10
        denom_small = pow(2, i)
        denom_large = pow(2,(i+1))

        if(segment_haversine_per_px >= starting_span):
            segment_mapbox_zoom = 10
            # st.write('segment_mapbox_zoom: ', segment_mapbox_zoom)
            break
        # st.write(zoom_level, starting_span*2/denom_small, segment_haversine_per_px, starting_span*2/denom_large)
        if(starting_span*2/denom_small > segment_haversine_per_px >= starting_span*2/denom_large):
            segment_mapbox_zoom = zoom_level
            # st.write('segment_mapbox_zoom: ', segment_mapbox_zoom)
            break

    # Plotly map from decoded polyline (that actually updates!)
    fig = go.Figure(go.Scattermapbox(
        lat=segment_path[:,0],
        lon=segment_path[:,1],
        mode='lines',
        opacity=0.75,
        line=dict(width=4)
        # text=[city_name],
    ))
    fig.update_layout(
        mapbox_style="outdoors",
        autosize=False,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=0
        ),
        hovermode=False,
        mapbox=dict(
            accesstoken=mapbox_token,
            bearing=0,
            center=dict(
                lat=(segment_path[0,0] + segment_path[-1,0])/2,
                lon=(segment_path[0,1] + segment_path[-1,1])/2
                # this centers the map at the geometric center of the area that the segment spans
            ),
            pitch=0,
            zoom=segment_mapbox_zoom
        ),
    )
    return fig

if w_person == 0:
    st.info('Please enter your weight in the sidebar')

else:
    grade = st.slider('Grade [%]', 0., 30., value=0., step=0.1) / 100
    st.plotly_chart(speedVsPowerPlot(w=w_total, G=grade), use_container_width=True)
    if grade != 0:
        st.plotly_chart(vamPlot(w=w_total, G=grade), use_container_width=True)
    

    st.header('Strava Segment Analysis')

    with open('segmentDict.txt', 'r') as infile:
        db = json.load(infile)
        sortedKeys = sorted(db, key=str.lower)
    selectedSegments = st.multiselect('Segment(s)', sortedKeys)

    for key in selectedSegments:
        # Display segment data
        st.subheader(db[key]['name'])
        st.write('https://www.strava.com/segments/' + str(db[key]['id']))
        st.write('Distance', round(db[key]['distance'] / 1000, 2), ' km = ', round(db[key]['distance'] / 1000 * 0.621, 2), ' mi')
        
        if(db[key]['total_elevation_gain'] == 0):
            totElevGain = round(db[key]['average_grade'] * db[key]['distance'] / 100)
        else:
            totElevGain = round(db[key]['total_elevation_gain'])
        st.write('Total elevation gain: ', totElevGain, ' m = ', round(totElevGain * 3.281), 'ft')
        
        st.write('Average grade: ', db[key]['average_grade'], '%')

        # Display segment on map
        st.plotly_chart(segmentMap(db, key), use_container_width=True)

        # Display time to finish plot
        if(db[key]['total_elevation_gain'] == 0):
            G = db[key]['average_grade'] / 100
            st.warning('The total elevation gain for this segment is missing in Strava\'s database so it has been calculated using the provided average grade. The resulting power calculation will be less accurate than for segments where the total elevation gain is a measured quantity.')
        else:
            # This should be more accurate
            G = db[key]['total_elevation_gain'] / db[key]['distance']
        st.plotly_chart(timeToFinishPlot(w=w_total, G=G, d=db[key]['distance']), use_container_width=True)

 


    ### Elevation profile
    # https://docs.mapbox.com/api/maps/#tilequery
    # https://docs.mapbox.com/help/tutorials/find-elevations-with-tilequery-api/
    # "Because the elevation data you want is included in the contour layer, you will need to parse the returned GeoJSON to isolate the features from the contour layer and find the highest elevation value."
    # mapbox API GET:/v4/{tileset_id}/tilequery/{lon},{lat}.json
    # payload = {'layers': 'contour', 'radius': '0', 'limit': '50', 'access_token': mapbox_token}
    # tileset_id = 'mapbox.mapbox-terrain-v2'
    # lat = 37.395614
    # lon = -122.247693
    # r = requests.get('https://api.mapbox.com/v4/'+tileset_id+'/tilequery/'+str(lon)+','+str(lat)+'.json', params=payload)
    # st.write(r.url)
    # r = r.json()

    # elev = []
    # for feature in r['features']:
    #     elev.append(feature['properties']['ele'])
    # st.write(elev)
    # point_elevation = max(elev)
    # st.write(point_elevation)
    # st.write(segment_path[0])

    # payload = {'layers': 'contour', 'radius': '0', 'limit': '50', 'access_token': mapbox_token}
    # tileset_id = 'mapbox.mapbox-terrain-v2'
    # i = 0
    # elevs = []
    # for coords in segment_path:
    #     if(i % 12 == 0):
    #         lat = coords[0]
    #         lon = coords[1]
    #         r = requests.get('https://api.mapbox.com/v4/'+tileset_id+'/tilequery/'+str(lon)+','+str(lat)+'.json', params=payload)
    #         r = r.json()

    #         elev = []
    #         for feature in r['features']:
    #             elev.append(feature['properties']['ele'])
    #         point_elevation = max(elev)*3.28
    #         elevs.append(point_elevation)
    #     i+=1

    # st.write(len(elevs))

    # st.area_chart(elevs)

    # # f2 = interp1d(elevs, kind='cubic')










    # st.write(r['features'])

    # st.write(r['features'])
    # st.write(r['features'][1]['properties'])
    # st.write('ele' in r['features'][1]['properties'])

    #
    #
    #

# with open('test_text.txt', 'r') as db:
#     st.write(db.readlines())