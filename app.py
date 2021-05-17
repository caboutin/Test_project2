# Import librairies 
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn import  linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Import data
df_2017 = pd.read_csv('IST_Civil_Pav_2017_Ene_Cons.csv')
df_2018 = pd.read_csv('IST_Civil_Pav_2018_Ene_Cons.csv')
df_holiday = pd.read_csv('holiday_17_18_19.csv')
df_meteo = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')

##### DATA PREPARARTION
# Clean data
df = pd.concat([df_2017,df_2018])
df['Date_start'] = pd.to_datetime(df['Date_start'])
df = df.set_index ('Date_start', drop = True)
df_meteo.rename(columns = {'yyyy-mm-dd hh:mm:ss': 'Date'}, inplace = True)
df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])
df_meteo = df_meteo.set_index ('Date', drop = True)
df_meteo = df_meteo.iloc[:186028,:] #Year 2019 removed
df_meteo = df_meteo.resample('H').mean() 
df = pd.merge(df, df_meteo, left_index=True, right_index=True, how ='outer')
df['Hour'] = df.index.hour
df['Day week'] = df.index.dayofweek
df['Date'] = df.index.date
df = df.set_index ('Date', drop = True)
df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
df_holiday = df_holiday.set_index ('Date', drop = True)
df_holiday = df_holiday.iloc[:28,:] #Year 2019 removed
df = pd.merge(df, df_holiday, left_index=True, right_index=True, how ='outer')
df['Holiday'] = df['Holiday'].fillna(0)  # NaN must be replaced by 0
df = df.reindex(columns=['Power_kW','temp_C','Hour','Day week','Holiday','HR','windSpeed_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day']) 
dfd = df.copy()
dfi = df.copy()
dfo = df.copy()

# 2017 vs 2018
dfd1 = dfd.iloc[:8761,:]
dfd2 = dfd.iloc[8761:,:]

# Remove outliers
index_with_nan = df.index[df.isnull().any(axis=1)]
df.drop(index_with_nan,0, inplace=True)
dfo = dfo[dfo['Power_kW'] >dfo['Power_kW'].quantile(0.25) ]
dfo  = dfo[dfo['temp_C'] >dfo['temp_C'].quantile(0.25) ]
dfo  = dfo[dfo['solarRad_W/m2'] >dfo['solarRad_W/m2'].quantile(0.25) ]

# Regression data
dfi=dfi.drop(columns=['temp_C','Day week','Holiday','HR','windSpeed_m/s','pres_mbar','rain_mm/h','rain_day','solarRad_W/m2'])
dfi['Power-1']=dfi['Power_kW'].shift(1)
dfi=dfi.dropna()
dfi  = dfi[dfi['Power_kW'] >dfi['Power_kW'].quantile(0.25) ]
X=dfi.values
Y=X[:,0]
X=X[:,1:]

# For our regression models
X_train, X_test, y_train, y_test = train_test_split(X,Y)
LR_model = linear_model.LinearRegression()
LR_model.fit(X_train,y_train)
y_pred_LR = LR_model.predict(X_test)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))
model_SVR = SVR(kernel='rbf')
model_SVR.fit(X_train_SVR,y_train_SVR.ravel())
y_pred_SVR = model_SVR.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)
DT_model = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
DT_model.fit(X_train, np.ravel(y_train))
y_pred_DT = DT_model.predict(X_test)
RF_model = RandomForestRegressor(bootstrap= True,min_samples_leaf= 3,n_estimators= 200, min_samples_split= 15,max_features= 'sqrt',max_depth= 20,max_leaf_nodes= None)
RF_model.fit(X_train, np.ravel(y_train))
y_pred_RF = RF_model.predict(X_test)
NN_model = MLPRegressor(hidden_layer_sizes=(10,30,30,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train,y_train.ravel())
GB_predictions =GB_model.predict(X_test)
XGB_model = XGBRegressor()
XGB_model.fit(X_train,y_train)
XGB_predictions =XGB_model.predict(X_test)
BT_model = BaggingRegressor()
BT_model.fit(X_train,y_train.ravel())
BT_predictions =BT_model.predict(X_test)


### FIGURES

# Fig 1 : Energy consumption : 2017 vs 2018
data = go.Figure()
data.add_trace(go.Scatter(x = dfd.index,y = dfd['Power_kW'],name = 'Total Power (kW)'))
data.add_trace(go.Scatter(x = dfd1.index,y = dfd1['Power_kW'],name = 'Power 2017 (kW) '))
data.add_trace(go.Scatter(x = dfd2.index,y = dfd2['Power_kW'],name = 'Power 2018 (kW) '))
data.update_layout(width=1000, height=350, colorway = ['blue','darkblue','cyan'], title ='Electricity consumption at the Civil Building of IST Lisboa (yearly):') 
data.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=list(
            [dict(label = 'Total',method = 'update',
                  args = [{'visible': [True, False, False]},
                          {'title': 'Total', 'showlegend':True}]),
             dict(label = '2017', method = 'update',
                  args = [{'visible': [False, True, False]}, 
                          {'title': '2017', 'showlegend':True}]),
               dict(label = '2018',method = 'update',
                    args = [{'visible': [False, False, True]},
                            {'title': '2018','showlegend':True}]),
            ])) ])

# Comparison  data with (fig1) and without outliers (figo):
fig1, figo = go.Figure(),go.Figure()
for column in df.columns.to_list():
    fig1.add_trace(go.Scatter(x = df.index,y = df[column],name = column))
    fig1.update_layout(width=700, height=500, colorway = ['darkblue','red','purple','pink','orange','cyan', 'grey','lightgreen','yellow','green', 'magenta']) 
fig1.update_layout(updatemenus=[go.layout.Updatemenu(active=0, x=0.57,y=1.2,buttons=list(
            [dict(label = 'All',method = 'update',
                  args = [{'visible': [True, True, True, True,True, True, True, True,True, True, True]},
                          {'title': 'All', 'showlegend':True}]),
             dict(label = 'Power', method = 'update',
                  args = [{'visible': [True, False, False, False, False, False, False, False, False, False, False]}, 
                          {'title': 'Power(kW)', 'showlegend':True}]),
               dict(label = 'Temperaure',method = 'update',
                    args = [{'visible': [False, True, False, False, False, False, False, False, False, False, False]},
                            {'title': 'Temperaure (°C)','showlegend':True}]),
               dict(label = 'Hour',method = 'update',
                    args = [{'visible': [False, False, True, False, False, False, False, False, False, False, False]},
                            {'title': 'Hour','showlegend':True}]),
               dict(label = 'Day Week',method = 'update',
                    args = [{'visible': [False, False, False, True,False, False, False, False, False, False, False]},
                            {'title': 'Day Week', 'showlegend':True}]),
               dict(label = 'Holiday',method = 'update',
                    args = [{'visible': [False, False, False,False, True, False, False, False, False, False, False]},
                            {'title': 'Holiday', 'showlegend':True}]),
               dict(label = 'Relative Humidity', method = 'update',
                    args = [{'visible': [False, False, False,False, False, True, False, False, False, False, False]},
                            {'title': 'Relative Humidity','showlegend':True}]),
                dict(label = 'Wind Speed',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False,True, False, False, False, False]},
                            {'title': 'Wind Speed (m/s)', 'showlegend':True}]),
                 dict(label = 'Pressure',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False,True, False, False, False]},
                            {'title': 'Pressure (mbar)','showlegend':True}]),
                 dict(label = 'Solar Radiation',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False,True, False, False]},
                            {'title': 'Solar Radiation (W/m2)','showlegend':True}]),
                 dict(label = 'Rain',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False ,False,True, False]},
                            {'title': 'Rain (mm/h)', 'showlegend':True}]),
                 dict(label = 'Rain day ',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False, False, False,True]},
                            {'title': 'Rain day','showlegend':True}]),
            ])) ])
for column in dfo.columns.to_list():
    figo.add_trace(go.Scatter(x = dfo.index,y = dfo[column],name = column))
    figo.update_layout(width=700, height=500, colorway = ['darkblue','red','purple','pink','orange','cyan', 'grey','lightgreen','yellow','green', 'magenta']) 
figo.update_layout(
updatemenus=[go.layout.Updatemenu(active=0, x=0.57,y=1.2,buttons=list(
            [dict(label = 'All',method = 'update',
                  args = [{'visible': [True, True, True, True,True, True, True, True,True, True, True]},
                          {'title': 'All', 'showlegend':True}]),
             dict(label = 'Power', method = 'update',
                  args = [{'visible': [True, False, False, False, False, False, False, False, False, False, False]}, 
                          {'title': 'Power(kW)', 'showlegend':True}]),
               dict(label = 'Temperaure',method = 'update',
                    args = [{'visible': [False, True, False, False, False, False, False, False, False, False, False]},
                            {'title': 'Temperaure (°C)','showlegend':True}]),
               dict(label = 'Hour',method = 'update',
                    args = [{'visible': [False, False, True, False, False, False, False, False, False, False, False]},
                            {'title': 'Hour','showlegend':True}]),
               dict(label = 'Day Week',method = 'update',
                    args = [{'visible': [False, False, False, True,False, False, False, False, False, False, False]},
                            {'title': 'Day Week', 'showlegend':True}]),
               dict(label = 'Holiday',method = 'update',
                    args = [{'visible': [False, False, False,False, True, False, False, False, False, False, False]},
                            {'title': 'Holiday', 'showlegend':True}]),
               dict(label = 'Relative Humidity', method = 'update',
                    args = [{'visible': [False, False, False,False, False, True, False, False, False, False, False]},
                            {'title': 'Relative Humidity','showlegend':True}]),
                dict(label = 'Wind Speed',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False,True, False, False, False, False]},
                            {'title': 'Wind Speed (m/s)', 'showlegend':True}]),
                 dict(label = 'Pressure',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False,True, False, False, False]},
                            {'title': 'Pressure (mbar)','showlegend':True}]),
                 dict(label = 'Solar Radiation',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False,True, False, False]},
                            {'title': 'Solar Radiation (W/m2)','showlegend':True}]),
                 dict(label = 'Rain',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False ,False,True, False]},
                            {'title': 'Rain (mm/h)', 'showlegend':True}]),
                 dict(label = 'Rain day ',method = 'update',
                    args = [{'visible': [False, False, False,False, False,  False, False, False, False, False,True]},
                            {'title': 'Rain day','showlegend':True}]),
            ])) ])

# Box figures to visualize outliers : 
Power ,Temperature,Hour,Day_week,Holiday,HR,WindSpeed, Pressure,SolarRad,Rain,Rain_Day =go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure(),go.Figure()
Power.add_trace(go.Box(x = df['Power_kW'],name = 'Power_kW',fillcolor='darkblue'))
Temperature.add_trace(go.Box(x = df['temp_C'],name = 'temp_C', fillcolor='red'))
Hour.add_trace(go.Box(x = df['Hour'],name = 'Hour', fillcolor='purple'))
Day_week.add_trace(go.Box(x = df['Day week'],name = 'Day week', fillcolor='pink'))
Holiday.add_trace(go.Box(x = df['Holiday'],name = 'Holiday', fillcolor='orange'))
HR.add_trace(go.Box(x = df['HR'],name = 'HR', fillcolor='cyan'))
WindSpeed.add_trace(go.Box(x = df['windSpeed_m/s'],name = 'windSpeed_m/s', fillcolor='grey'))
Pressure.add_trace(go.Box(x = df['pres_mbar'],name = 'pres_mbar', fillcolor='lightgreen'))
SolarRad.add_trace(go.Box(x = df['solarRad_W/m2'],name = 'solarRad_W/m2', fillcolor='yellow'))
Rain.add_trace(go.Box(x = df['rain_mm/h'],name = 'rain_mm/h', fillcolor='green'))
Rain_Day.add_trace(go.Box(x = df['rain_day'],name = 'rain_day', fillcolor='magenta'))
fig2 = ['Power' ,'Temperature','Hour','Day_week','Holiday','HR','WindSpeed', 'Pressure','SolarRad','Rain','Rain_Day']

# Clustering and feature selection graphs :
img1,img2, img3, img4, img5 = 'cluster1.png', 'cluster2.png','cluster3.png','cluster4.png', 'cluster5.png'
img1,img2, img3, img4 , img5= base64.b64encode(open(img1, 'rb').read()).decode('ascii'),base64.b64encode(open(img2, 'rb').read()).decode('ascii'),base64.b64encode(open(img3, 'rb').read()).decode('ascii'),base64.b64encode(open(img4, 'rb').read()).decode('ascii'),base64.b64encode(open(img5, 'rb').read()).decode('ascii')
feat1,feat2, feat3= 'feat1.png', 'feat2.png','feat3.png'
feat1,feat2, feat3= base64.b64encode(open(feat1, 'rb').read()).decode('ascii'),base64.b64encode(open(feat2, 'rb').read()).decode('ascii'),base64.b64encode(open(feat3, 'rb').read()).decode('ascii')

# Regression model figures :
reg11,reg12 = go.Figure(),go.Figure()
reg11.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg11.add_trace(go.Scatter(y = y_pred_LR[1:200],mode='lines',name='model',line_color = 'pink'))
reg12.add_trace(go.Scatter(x=y_test,y=y_pred_LR,mode='markers',marker_color = 'pink'))
reg21,reg22 = go.Figure(),go.Figure()
reg21.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg21.add_trace(go.Scatter(y = y_pred_SVR2[1:200],mode='lines',name='model',line_color = 'magenta'))
reg22.add_trace(go.Scatter(x=y_test,y=y_pred_SVR2,mode='markers',marker_color = 'magenta'))
reg31,reg32 = go.Figure(),go.Figure()
reg31.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg31.add_trace(go.Scatter(y = y_pred_DT[1:200],mode='lines',name='model',line_color = 'red'))
reg32.add_trace(go.Scatter(x=y_test,y=y_pred_DT,mode='markers',marker_color = 'red'))
reg41,reg42 = go.Figure(),go.Figure()
reg41.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg41.add_trace(go.Scatter(y = y_pred_RF[1:200],mode='lines',name='model',line_color = 'orange'))
reg42.add_trace(go.Scatter(x=y_test,y=y_pred_RF,mode='markers',marker_color = 'orange'))
reg51,reg52 = go.Figure(),go.Figure()
reg51.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg51.add_trace(go.Scatter(y = y_pred_NN[1:200],mode='lines',name='model',line_color = 'yellow'))
reg52.add_trace(go.Scatter(x=y_test,y=y_pred_NN,mode='markers',marker_color = 'yellow'))
reg61,reg62 = go.Figure(),go.Figure()
reg61.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg61.add_trace(go.Scatter(y = GB_predictions[1:200],mode='lines',name='model',line_color = 'cyan'))
reg62.add_trace(go.Scatter(x=y_test,y=GB_predictions,mode='markers',marker_color = 'cyan'))
reg71,reg72 = go.Figure(),go.Figure()
reg71.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg71.add_trace(go.Scatter(y = XGB_predictions[1:200],mode='lines',name='model',line_color = 'blue'))
reg72.add_trace(go.Scatter(x=y_test,y=XGB_predictions,mode='markers',marker_color = 'blue'))
reg81,reg82 = go.Figure(),go.Figure()
reg81.add_trace(go.Scatter(y = y_test[1:200], mode='lines',name='test',line_color ='black'))
reg81.add_trace(go.Scatter(y = BT_predictions[1:200],mode='lines',name='model',line_color = 'green'))
reg82.add_trace(go.Scatter(x=y_test,y=BT_predictions,mode='markers',marker_color = 'green'))

# Performance metrics
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
LR = [MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR]
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
SVR = [MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR]
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
DT = [MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT]
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
RF = [MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF]
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
NN = [MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN]
MAE_GB=metrics.mean_absolute_error(y_test,GB_predictions) 
MSE_GB=metrics.mean_squared_error(y_test,GB_predictions)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,GB_predictions))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
GB = [MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB]
MAE_XGB=metrics.mean_absolute_error(y_test,XGB_predictions) 
MSE_XGB=metrics.mean_squared_error(y_test,XGB_predictions)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,XGB_predictions))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
XGB = [MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB]
MAE_BT=metrics.mean_absolute_error(y_test,BT_predictions) 
MSE_BT=metrics.mean_squared_error(y_test,BT_predictions)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,BT_predictions))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
BT = [MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT]
LR,SVR,DT,RF,NN,GB,XGB,BT = np.around(LR , decimals = 4),np.around(SVR , decimals = 4),np.around(DT , decimals = 4),np.around(RF , decimals = 4),np.around(NN , decimals = 4),np.around(GB , decimals = 4),np.around(XGB , decimals = 4),np.around(BT , decimals = 4)
PM = ['MAE', 'MSE', 'RMSE','cvRMSE']
Pam = [PM,LR,SVR,DT,RF,NN,GB,XGB,BT]

# Performance metrics table
colorhead=['white','salmon','magenta','red','orange','yellow','cyan','blue','green']
colorfont=['white','lightpink','violet','tomato','peachpuff','lightyellow','lightcyan','lightblue','lightgreen']
Categories = ['Performance Metrics',"Linear Regression","Support Vector Regression","Decision Trees","Random Forest",'Neural Networks', 'Gradient Boosting', 'Extreme Gradient Boosting', 'Boostrapping'] 
tablePam = go.Figure(data=[go.Table(columnwidth = [800,800,800,800,800,800,800,800],header = dict(values = Categories,line_color='darkslategray',fill_color=colorhead,align=['left','center'],height=40),cells=dict( values=Pam,line_color='darkslategray',fill=dict(color= colorfont), align=['left', 'center'],))])

# Performance metrics graph :
pm = 'pm.png'
pm = base64.b64encode(open(pm, 'rb').read()).decode('ascii')


### APP :
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
     html.H1(children='Project 2:'),  html.H6(children='The objective is to develop a dashboard of the main results of Project 1.'), 'PROJECT 1 : Forecast of energy demand using machine learning.  The objective was to develop a model to forecast electricity consumption in the buildings of IST : Civil Building.',
      dcc.Graph(figure= data),
     'Here you can see the model building steps : Data analysis, Clustering , Feature Selection and Regression.',
    dcc.Tabs(id='tabs-1',value='Exploratory data analysis.',children=[dcc.Tab(label='Exploratory data analysis',value='tab-1',
                children=[ html.Div("First, you can observe our data and observe outliers. Then, you can compare our data with and without outliers."),
                    dcc.Tabs(id='tabs-2', value='Observe data',
                        children=[
                            dcc.Tab(label='Observe data', value='tab-1-1', children=[html.H3(children='Observe data:'),html.H6(children='Check for outliers visually (and have a look to the notes).'),
                              html.Div([
        html.Div([        
           dcc.Graph(figure= fig1)
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='plot'),

        html.Div([
            dcc.Dropdown(
                id='variables',
                options=[{'label': i , 'value': i} for i in fig2 ],
                value=fig2[0]
            )
        ])
        ], className="six columns"),
    ], className="row")  ,html.H5(children='Notes: '), html.H6(children='- Power: '),'Power values above 450kW are rare but not necessarily outliers. Power values below 75kW are probably "inaccurate" measurements.  We can notice that a lot of data is missing around October/November 2018  (because we removed NaN in the Temperature column).' ,                                               
        html.H6(children='- Temperature : '), 'In summer, the temperature can reach these values which are considered here as outliers. The highest values are indeed those reached in summer.  We can notice that a lot of data is missing around October/November 2018  (NaN).',              
        html.H6(children='- Solar radiation : '), 'High values are considered to be outliers. We can notice again that a lot of data is missing around October/November 2018, and also around March and October 2017.'

    ]),
     dcc.Tab(label='Outliers removed', value='tab-1-2', children=[html.H3(children='Outliers Removed:'),html.H6(children='The first graph presents our data and the second one, our data without ouliers. '),
     html.Div([
        html.Div([
            dcc.Graph(figure=fig1)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=figo)
        ], className="six columns"),
    ], className="row")  ,
   html.H5(children='Notes: '),  'Power values below 100 kW were removed. Temperature values below 10°C were removed.  '   ] ),   
                        ])]),
            dcc.Tab(label='Clustering',value='tab-2',
                children=[ html.Div("Here you can observe the different clusters. We choose a number of clusters of 3."),
                    dcc.Tabs(id='tabs-3',value='Clustering',
                        children=[
                            dcc.Tab(label='Cluster Analysis', value='tab-2-1', children=[html.H3(children='Cluster Analysis:'), html.H6(children='Temperature and Day Week indicators:'),
                                   'We can not really observe a correlation between Power and Temperature and between Power and Week day. ',
                                       html.Img(src='data:image/png;base64,{}'.format(img1)),
                                       html.H6(children='Hour indicator:'),
                                    'It seems one cluster applies to the hours from 0h to 8h and from around 18h to 24h (what we can called "extra" hours). The other clusters relate to the "hours of the day", from 8h to 18h. (One of them relate more to hours from 8h to 10h and 16h to 18h and the other one, 10h to 16h). So, it looks like Hours is one of the criteria. As we could observe on the Power/Hour diagramme, the Week day/Hour diagramme corroborates that one of the cluster is relating to "extra" hours.',
                                     html.Img(src='data:image/png;base64,{}'.format(img2)), 
                                     html.H6(children='Solar radiation indicator:'),
                                    'It is clear that solarRad_W/m2 is a criteria for clustering. A cluster (which related to the "extra" hours) corresponds to the lowest solar radiation (below 200 W/m2). Then the second one is about the radiations between 200 and 600 W/m2. Last one relates to the highest solar radiation (above 600 W/m2). Solar radiation is a very good indicator of the activity/energy of the building.  In fact, people arrive when the sun is going up and leave when the sun is going down. We can by the way see the correlation between this indicator and "Hour".',
                                    html.Img(src='data:image/png;base64,{}'.format(img3)), 
                                    html.H6(children='Hour / Power vs Day week or vs Solar Radiation:'),
                                    html.Img(src='data:image/png;base64,{}'.format(img4))]),
                            dcc.Tab(label='Identifying daily patterns', value='tab-2-2', children=[ html.H3(children='Identifying daily patterns:'),
                                    html.Img(src='data:image/png;base64,{}'.format(img5))])
                        ])]),
             dcc.Tab(label='Feature Selection',value='tab-3',
                children=[ html.Div("Here we are going to select the best criteria. To do it, we are going to use three methods : Filter method, Wrapper method and Ensemble method."),
                    dcc.Tabs(id='tabs-4',value='Feature Selection:',
                        children=[
                            dcc.Tab(label='Filter Method', value='tab-3-1', children=[html.H3(children='Filter Method:'), html.H6(children='kBest:'),'The best criteria is Power-1, then solarRad_W/m2 and finally Day week (if we consider k=3).',
                                    html.Img(src='data:image/png;base64,{}'.format(feat1)), ]),
                            dcc.Tab(label='Wrapper Method', value='tab-3-2', children=[ html.H3(children='Wrapper Method:'),html.H6(children='Recursive Feature Elimination (RFE) : LinearRegression Model as Estimator:'), 'Here best criteria is Holiday, then Day week and then rain_mm/h.',
                                    html.Img(src='data:image/png;base64,{}'.format(feat2))]),
                            dcc.Tab(label=' Ensemble Method ', value='tab-3-3', children=[ html.H3(children=' Ensemble Method: '), 'Best criteria is Power-1, then Hour and then solarRad_W/m2.',
                                    html.Img(src='data:image/png;base64,{}'.format(feat3)) ])
                        ])]),
             
             
             dcc.Tab(label='Regression & Conclusion',value='tab-4',
                children=[ html.Div("Here we will see which regression model is the best one."),
                    dcc.Tabs(id='tabs-5',value='Regression & conclusin',
                        children=[
                            dcc.Tab(label='Regression Models', value='tab-4-1', children=[html.H3(children='Regression Models:'), html.H6(children='Here are differents regression models. We will see which one in the best thanks to performance metrics.'), 'From the clustering and feature selection parts, we can choose Power-1 and Hour (and SolarRad if we want three inlets) as inlets for our model. After several tests, the best models have been found with only two inlets : Power-1 and Hour.',
                            html.H6(children='Linear Regression:'),             
                                    
        html.Div([                      
        html.Div([
            dcc.Graph(figure=reg11)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg12)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Support Vector Regression:'),
     html.Div([                           
        html.Div([
            dcc.Graph(figure=reg21)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg22)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Regression Decision Trees:'),
     html.Div([                           
        html.Div([
            dcc.Graph(figure=reg31)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg32)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Random Forest:'),
     html.Div([                           
        html.Div([
            dcc.Graph(figure=reg41)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg42)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Neural Network:'),
     html.Div([                           
        html.Div([
            dcc.Graph(figure=reg51)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg52)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Gradient Boosting:'),html.Div([                           
        html.Div([
            dcc.Graph(figure=reg61)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg62)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Extreme Gradient Boosting:'),html.Div([                           
        html.Div([
            dcc.Graph(figure=reg71)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg72)
        ], className="six columns"),
    ], className="row"),
     html.H6(children='Boostrapping:'),html.Div([                           
        html.Div([
            dcc.Graph(figure=reg81)
        ], className="six columns"),

        html.Div([
            dcc.Graph(figure=reg82)
        ], className="six columns"),
    ], className="row")
                                   ]),
                            dcc.Tab(label='Performance Metrics & Conclusion', value='tab-4-2', children=[ html.H3(children='Performance Metrics:'),
          html.H5(children='Conclusion: '),  'The best model seems to be the Random Forest one. (Gradient Boosting and Extreme Gradient Boosting are also good ones). These table and graph illustrate it :',
        dcc.Graph(figure=tablePam)  ,html.Img(src='data:image/png;base64,{}'.format(pm))
           
           ])
                        ])])
        ]) ])
 
    
@app.callback(
    Output('plot', 'figure'),
    [Input('variables', 'value')])

def update_graph(fig_name):
    if fig_name == 'Power':
        return Power
    if fig_name == 'Temperature':
        return Temperature
    if fig_name == 'Holiday':
        return Holiday
    if fig_name == 'HR':
        return HR
    if fig_name == 'SolarRad':
        return SolarRad
    if fig_name == 'Rain':
        return Rain
    if fig_name == 'Rain_Day':
        return Rain_Day
    if fig_name == 'Day_week':
        return Day_week
    if fig_name == 'Pressure':
        return Pressure
    if fig_name == 'WindSpeed':
        return WindSpeed
    if fig_name == 'Hour':
        return  Hour 




if __name__ == '__main__':
    app.run_server(debug=False)
