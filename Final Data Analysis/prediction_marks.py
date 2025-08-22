import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.title(":open_book: Student Score Prediction")

df1=pd.read_csv('StudentsPerformance.csv')

st.header('Please enter the details :point_down:')
df1.columns=df1.columns.str.replace(".","_") #here we've replaced "." by "_" should use str before function to use string functions in pandas
df1.columns=df1.columns.str.replace("/","_") #here we've replaced "/" by "_" 
df1.columns=df1.columns.str.replace(" ","_") #here

cat_col=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']

gender_map = {"male": 0, "female": 1}
race_map={"group B":0,"group C":1,"group A":2,"group D":3,"group E":4}
parent_map={"bachelor's degree":0,"some college":1,"master's degree":2, "associate's degree":3,"high school":4,"some high school":5}
lunch_map={"standard":0,"free/reduced":1}
prep_map={"none":0,"completed":1}

df1["gender"] = df1["gender"].map(gender_map)
df1["race_ethnicity"] = df1["race_ethnicity"].map(race_map)
df1["parental_level_of_education"] = df1["parental_level_of_education"].map(parent_map)
df1["lunch"] = df1["lunch"].map(lunch_map)
df1["test_preparation_course"] = df1["test_preparation_course"].map(prep_map)

#st.header('Dataset Overview')
#df1.head()

X = df1.drop(['math_score','reading_score','writing_score'],axis=1)
y_maths = df1['math_score']
y_reading = df1['reading_score']
y_writing = df1['writing_score']

X_train, X_test, y_train_maths, y_test_maths = train_test_split(X, y_maths, test_size=0.33, random_state=42)
X_train, X_test, y_train_reading, y_test_reading = train_test_split(X, y_reading, test_size=0.33, random_state=42)
X_train, X_test, y_train_writing, y_test_writing = train_test_split(X, y_writing, test_size=0.33, random_state=42)

rfr_m = RandomForestRegressor(min_samples_leaf=20)
rfr_r = RandomForestRegressor(min_samples_leaf=20)
rfr_w = RandomForestRegressor(min_samples_leaf=20)

rfr_m.fit(X_train,y_train_maths)
pred_maths = rfr_m.predict(X_test)
mae1=mean_absolute_error(y_test_maths,pred_maths)

rfr_r.fit(X_train,y_train_reading)
pred_reading = rfr_r.predict(X_test)
mae2=mean_absolute_error(y_test_reading,pred_reading)

rfr_w.fit(X_train,y_train_writing)
pred_writing = rfr_w.predict(X_test)
mae3=mean_absolute_error(y_test_writing,pred_writing)

gender_map = {"Male": 0, "Female": 1}
gender = st.selectbox(
    "Enter gender of the student",
    ("Male", "Female"),
)
n1=gender_map[gender] 

race_map={"group B":0,"group C":1,"group A":2,"group D":3,"group E":4}
race = st.selectbox(
    "Enter race/ethnicity group of the student",
    ("group B", "group C","group A","group D","group E"),
)
n2=race_map[race]

parent_map={"bachelor's degree":0,"some college":1,"master's degree":2, "associate's degree":3,"high school":4,"some high school":5}
pqual = st.selectbox(
    "Enter parents qualification of the student",
    ("bachelor's degree", "some college","master's degree","associate's degree","high school","some high school"),
)

n3=parent_map[pqual]

lunch_map={"standard":0,"free/reduced":1}
pqual = st.selectbox(
    "Enter lunch menu of the student",
    ("standard", "free/reduced"),
)
n4=lunch_map[pqual]

prep_map={"none":0,"completed":1}
prep = st.selectbox(
    "Have they selected test preparation course?",
    ("none", "completed"),
)
n5=prep_map[prep]

sample=[[n1,n2,n3,n4,n5]]

c1,c2,c3=st.columns(3)
if c1.button('Predict maths score'):
    t1=rfr_m.predict(sample)
    c1.subheader("Predicted Maths score")
    c1.subheader(t1)

if c2.button('Predict reading score'):
    t2=rfr_r.predict(sample)
    c2.subheader("Predicted reading score")
    c2.subheader(t2)

if c3.button('Predict writing score'):
    t3=rfr_w.predict(sample)
    c3.subheader("Predicted writing score")
    c3.subheader(t3)
st.divider()
if st.button('Thank you Visit Again',use_container_width=True):
    st.balloons()
