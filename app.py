import streamlit as st
#import streamlit as st
import sklearn
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#df = pd.read_pickle(open(r"C:\Users\ajay2\Desktop\ML\Laptaop_price_predictor\df.pkl",'rb'))
df=pd.read_pickle(open("df.pkl",'rb'))
# print(df.shape)
# print(df.head())
x=df.drop(columns=['Price'])
y=np.log(df['Price'])
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.15,random_state=2)
 
#Linear regresion
catego_features_idx=[0,1,7,9,10]
step_lr = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False, drop='first', categories='auto' ),catego_features_idx)
],remainder='passthrough')

lr = LinearRegression()

pipe_lr = Pipeline([
    ('step1',step_lr),
    ('step2',lr)
])

pipe_lr.fit(X_train,Y_train)

Y_pred = pipe_lr.predict(X_test)

# print('R2 score',r2_score(Y_test,Y_pred))
# print('MAE',mean_absolute_error(Y_test,Y_pred))

#KNN

step_KNN = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False, drop='first', categories='auto' ),catego_features_idx)
],remainder='passthrough')

KNN = KNeighborsRegressor(n_neighbors=3)

pipe_KNN = Pipeline([
    ('step1',step_KNN),
    ('step2',KNN)
])

pipe_KNN.fit(X_train,Y_train)

Y_pred = pipe_KNN.predict(X_test)

# print('R2 score',r2_score(Y_test,Y_pred))
# print('MAE',mean_absolute_error(Y_test,Y_pred))

#DECISION TREE

step_DT = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False, drop='first', categories='auto' ),catego_features_idx)
],remainder='passthrough')

DT = DecisionTreeRegressor(max_depth=8)

pipe_DT = Pipeline([
    ('step1',step_DT),
    ('step2',DT)
])

pipe_DT.fit(X_train,Y_train)

Y_pred = pipe_DT.predict(X_test)

# print('R2 score',r2_score(Y_test,Y_pred))
# print('MAE',mean_absolute_error(Y_test,Y_pred))

# RANDOM FOREST

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False, drop='first', categories='auto' ),catego_features_idx)
],remainder='passthrough')

rd = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe_rd = Pipeline([
    ('step1',step1),
    ('step2',rd)
])

pipe_rd.fit(X_train,Y_train)

Y_pred = pipe_rd.predict(X_test)

print('R2 score',r2_score(Y_test,Y_pred))
print('MAE',mean_absolute_error(Y_test,Y_pred))


st.title("Predict Price of Your Dream Laptop ")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop (in Kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS (Instrusion Prevention System)',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size (in Inchess)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

#hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('Oprating System',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,ssd,gpu,os])

    query = query.reshape(1,11)
    st.title("The predicted price of this configuration is (in Rs.)" + str(int(np.exp(pipe_rd.predict(query)[0]))))
