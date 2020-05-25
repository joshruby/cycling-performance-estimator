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
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests_oauthlib import OAuth2Session
import shelve
import polyline
import plotly.graph_objects as go
from haversine import haversine, Unit

mapbox_token = 'pk.eyJ1IjoianJydWJ5IiwiYSI6ImNrOWtrMDU3czF2dTkzZG53Nmw2NDdneTMifQ.zzXEhr0Z1biR2pydOFco8A'


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

# st.header('Cycling Performance Estimator')


# data = p_legs_table(w=150, G=0.080, v_center=7, units='imperial')

# fig, ax = plt.subplots(constrained_layout=True)
# plt.title('Weight of Rider with Gear = 150 lbs, Grade = 8.0%')

# ax.plot(data[:, 1], data[:, 0])
# ax.set_xlabel("Power", fontsize=14)
# ax.set_ylabel("Speed", fontsize=14)

# # These methods are only used to set up the double-x axis in the plot
# # Kind of annoying that this is how you do it
# def power_to_powerperkg(x):
#     return x / (150 / 2.20462) # if units = imperial
#     return x / 150 # if units = metric

# def powerperkg_to_power(x):
#     return x * (150 / 2.20462) # if units = imperial
#     return x * 150 # if units = metric

# # Second x axis that lives above the top line
# secax = ax.secondary_xaxis('top', functions=(power_to_powerperkg, powerperkg_to_power))
# secax.set_xlabel('Power/kg')

# # Second y axis that lives on the right of plot
# ax2=ax.twinx()
# ax2.plot(data[:, 1], data[:, 2])
# ax2.set_ylabel("VAM", fontsize=14)

# plt.grid(True)
# # plt.show()

# st.write(fig)

# st.subheader('Time to finish OLH (Strava ID 8109834)')
# data = p_legs_table(150, 0.08, 7, units = 'imperial')
# ttf = 2.98 / data[:, 0] * 60
# data = np.column_stack((data, ttf))
# data = pd.DataFrame(data,
#                     columns = ['Speed [m/s]', 'Power [W]', 'VAM [m/hr]', 'Time [min]']
# )
# st.table(data)


### OAuth 2
# st.subheader('https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html#web-application-flow')

# client_id = '45520'
# client_secret = 'ca22713c82948cee145911075f42ea2a830cf9a5'
# redirect_uri = 'http://localhost:8501'
# response_type = 'code'
# scope = 'read'
# approval_prompt = 'force'

# oauth = OAuth2Session(client_id, 
#         redirect_uri=redirect_uri, 
#         scope=scope)

# authorization_url, state = oauth.authorization_url(
#         'https://www.strava.com/oauth/authorize?',
#         # access_type and prompt are Strava-specific extra parameters.
#         response_type=response_type,
#         approval_prompt=approval_prompt)

# st.write('Please go to %s and authorize access.' % authorization_url)

#
#
#


###
# from oauthlib.oauth2 import BackendApplicationClient
# client = BackendApplicationClient(client_id=client_id)
# oauth = OAuth2Session(client=client)
# token = oauth.fetch_token(token_url='https://www.strava.com/api/v3/oauth/token', client_id=client_id,
#         client_secret=client_secret)


#
#
#


###

# from oauthlib.oauth2 import BackendApplicationClient
# from requests.auth import HTTPBasicAuth
# auth = HTTPBasicAuth(client_id, client_secret)
# client = BackendApplicationClient(client_id=client_id)
# oauth = OAuth2Session(client=client)
# token = oauth.fetch_token(token_url='https://www.strava.com/api/v3/oauth/8126fdfa6dd6888786c3d8d9a8ce055af5818f80', auth=auth)

#
#
#


###
# import asyncio

# import streamlit as st
# from httpx_oauth.clients.google import GoogleOAuth2

# st.title("Google OAuth2 flow")

# "## Configuration"

# client_id = st.text_input("Client ID")
# client_secret = st.text_input("Client secret")
# redirect_uri = st.text_input("Redirect URI", "http://localhost:8501/redirect")

# if client_id and client_secret and redirect_uri:
#     client = GoogleOAuth2(client_id, client_secret)
# else:
#     client = None
    
# "## Authorization URL"

# async def write_authorization_url():
#     authorization_url = await client.get_authorization_url(
#         redirect_uri,
#         scope=["profile", "email"],
#         extras_params={"access_type": "offline"},
#     )
#     st.write(authorization_url)

# if client:
#     asyncio.run(write_authorization_url())
# else:
#     "Waiting client configuration..."

# "## Callback"

# if client:
#     code = st.text_input("Authorization code")
# else:
#     code = None
#     "Waiting client configuration..."

# "## Access token"

# async def write_access_token(code):
#     token = await client.get_access_token(code, redirect_uri)
#     st.write(token)

# if code:
#     asyncio.run(write_access_token(code))
# else:
#     "Waiting authorization code..."
#
#
#


###
# https://requests-oauthlib.readthedocs.io/en/latest/examples/github.html

# client_id = '45520'
# client_secret = 'ca22713c82948cee145911075f42ea2a830cf9a5'

# # OAuth endpoints given in the GitHub API documentation
# authorization_base_url = 'https://www.strava.com/oauth/authorize'
# token_url = 'https://www.strava.com/oauth/token'


# from requests_oauthlib import OAuth2Session
# strava = OAuth2Session(client_id)

# # Redirect user to GitHub for authorization
# authorization_url, state = strava.authorization_url(authorization_base_url)
# st.write('Please go here and authorize,', authorization_url)

# # Get the authorization verifier code from the callback url
# redirect_response = st.text_input('Paste the full redirect URL here:')

# # Fetch the access token
# github.fetch_token(token_url, client_secret=client_secret,
#         authorization_response=redirect_response)

# # Fetch a protected resource, i.e. user profile
# r = github.get('https://api.github.com/user')
# print r.content

#
#
#


###
# https://www.reddit.com/r/Strava/comments/az5gcv/strava_api_token_exchange/
# Replace the client id in this link with mine: https://strava.com/oauth/authorize?response_type=code&client_id=45520&redirect_uri=https://localhost/callback&scope=read_all
# Paste the link in the browser and authorize
# Then copy and past the returned URL
# Which looks like this:
# https://localhost/callback?state=&code=14987cbasdfasfsfsfsfdasdfd9f91917c1&scope=read,read_all 
# The code is what will be used to request a temporary access token
# The token is what needs to be passed with each API get request 



#
#
#


### 
# https://stackoverflow.com/questions/52880434/problem-with-access-token-in-strava-api-v3-get-all-athlete-activities
# After authorizing, Strava sends the user back to the redirect_uri. The script needs to read the "code" and "scope" from the url of the redirect_uri. I can't figure out how to do this programmatically.

# st.subheader('https://stackoverflow.com/questions/52880434/problem-with-access-token-in-strava-api-v3-get-all-athlete-activities')

# client_id = '45520'
# redirect_uri = 'http://localhost:8501'
# response_type = 'code'
# scope = 'read'

# authorizationPage = 'https://www.strava.com/oauth/authorize?'+'client_id='+client_id+'&' + 'response_type='+response_type+'&' + 'redirect_uri='+redirect_uri+'&' + 'scope='+scope+'&' + 'approval_prompt=force'

# st.write('Please go here to authorize: ' + authorizationPage)

# authorizationRedirect = st.text_input('Enter page url', value='', type='default')

#
#
#







### Load segments
# st.subheader('Storing segment data with shelve so users don"t need to authenticate (data loaded using my personal access token)')

# segmentName = 'Kings Mountain (Tripp to Skyline)'
# segmentID = 611413

# headers = {"Authorization": "Bearer 4436257166e049cbbe20c49ac7bf451c12bdf4a8"}
# call = 'https://www.strava.com/api/v3/segments/'+str(segmentID)
# # st.write(call)
# r = requests.get(call, headers=headers)
# r = r.json()
# # st.write(r)

# with shelve.open('segments') as db:
#     db[segmentName] = r
#     # st.subheader('Shelve value')
#     # st.write(db[segmentName])
#     # del db['PSkyline - Old La Honda to Page Mill']
#     st.subheader('segments.db keys')
#     st.write(list(db.keys()))

#
#
#

w_person = st.sidebar.number_input('Rider Weight [kg]')
w_gear = st.sidebar.number_input('Gear Weight [kg]')
w = w_person + w_gear

with shelve.open('segments') as db:
    sortedKeys = sorted(db, key=str.lower)
selectedSegments = st.sidebar.multiselect('Segment(s)', sortedKeys, default='Old La Honda (Bridge to Mailboxes)')
# st.write(selectedSegments)

for key in selectedSegments:
    with shelve.open('segments') as db:
        # st.write(db[key])
        st.subheader(db[key]['name'])
        st.write('https://www.strava.com/segments/' + str(db[key]['id']))
        # st.write(db[key]['total_elevation_gain'])
        if(db[key]['total_elevation_gain'] == 0):
            st.write('Total elevation gain: **NO DATA** (blame Strava!)')
        else:
            st.write('Total elevation gain: ', db[key]['total_elevation_gain']*3.281, '[ft]')
        st.write('Average grade: ', db[key]['average_grade'], '[%]')

        # returns a list of decoded (lat, lon) tuples
        segment_path = polyline.decode(db[key]['map']['polyline'])
        segment_path = np.array(segment_path)
        segment_haversine = haversine(db[key]['start_latlng'], db[key]['end_latlng'], unit=Unit.METERS)
        # st.write('Haversine: ', segment_haversine)
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
                # this centers the map on the center of the segment
            ),
            pitch=0,
            zoom=segment_mapbox_zoom
        ),
    )

    st.plotly_chart(fig, use_container_width=True)