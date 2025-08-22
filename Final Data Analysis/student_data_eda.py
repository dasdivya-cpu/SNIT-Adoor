import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as mat
import plotly.express as px

st.set_page_config(page_title="Student Data Analysis", page_icon="	:roller_coaster:",layout="wide")

st.title(":male-teacher: Student Data Analysis :male-student:")

df=pd.read_csv('StudentsPerformance.csv')
st.dataframe(df.head())
st.divider()
df.columns=df.columns.str.replace(".","_") #here we've replaced "." by "_" should use str before function to use string functions in pandas
df.columns=df.columns.str.replace("/","_") #here we've replaced "/" by "_" 
df.columns=df.columns.str.replace(" ","_") #here we replaced " " by "_" 

st.subheader("New dataframe")
st.dataframe(df.head())
st.divider()
#EDA
col1,col2=st.columns(2)
col1.header(":books: Statistical Summary")
col1.table(df.describe())

col2.header(":bookmark_tabs: Presence of null values")
col2.write(df.isnull().sum())
st.divider()

st.header(":chart_with_downwards_trend: Visual Representation of Data")
c1,c2=st.columns(2)
c1.subheader("What is the gender distribution in Dataset")
c1.bar_chart(df['gender'].value_counts(),color=["#487A3E"])

c2.subheader("What is the race/ethnicity distribution in Dataset")
c2.bar_chart(df['race_ethnicity'].value_counts(),color=["#008080"])

c1.subheader("What is the count of parental level of education in Dataset")
c1.bar_chart(df['parental_level_of_education'].value_counts(),color=["#008000"])

c2.subheader("Count of people took test preparation course")
c2.bar_chart(df['test_preparation_course'].value_counts(),color=["#808000"])
st.divider()
st.header(":bar_chart: Distribution of Marks")
c3,c4,c5=st.columns(3)
c3.subheader(" Distribution of Maths Score ")
fig1 = px.histogram(df, x='math_score',color_discrete_sequence=['indianred'])
#sns.displot(df['writing_score'])
c3.plotly_chart(fig1)


c4.subheader("Distribution of Reading Score")
fig2 = px.histogram(df, x='reading_score',color_discrete_sequence=['red'])
#sns.displot(df['writing_score'])
c4.plotly_chart(fig2)

c5.subheader("Distribution of Writing Score")
fig3 = px.histogram(df, x='writing_score',color_discrete_sequence=['orange'])
#sns.displot(df['writing_score'])
c5.plotly_chart(fig3)
st.divider()


#fig4, ax = plt.subplots(figsize=(5, 5))
c6,c7,c8,c9=st.columns(4)
c6.header("Relation of Scores")
matrix=df[['math_score','reading_score','writing_score']].corr()
fig4=px.imshow(matrix,
              labels=dict(x="Marks", y="Marks"),
                x=['Maths', 'Reading', 'Writing'],
                y=['Maths', 'Reading', 'Writing'])
c6.plotly_chart(fig4)

c7.header("Relation of Maths and Reading Scores")
#matrix=df[['math_score','reading_score','writing_score']].corr()
fig5= px.scatter(df, x="math_score", y="reading_score", color="gender", title="Relation of Maths and Reading Scores")
c7.plotly_chart(fig5)

c8.header("Relation of Reading and Writing Scores")
#matrix=df[['math_score','reading_score','writing_score']].corr()
fig6= px.scatter(df, x="writing_score", y="reading_score", color="gender", title="Relation of Writing and Reading Scores")
c8.plotly_chart(fig6)

c9.header("Relation of Maths and Writing Scores")
#matrix=df[['math_score','reading_score','writing_score']].corr()
fig7= px.scatter(df, x="math_score", y="writing_score", color="gender", title="Relation of Maths and Writing Scores")
c9.plotly_chart(fig7)
st.divider()

st.header('Comparison of Marks based on different Factors :chart_with_upwards_trend:')
st.divider()
marks=df[['math_score','reading_score','writing_score']]
c10,c11=st.columns(2)
c10.subheader('Based on Gender')
ax1=df.groupby(by='gender')[['math_score', 'reading_score','writing_score']].mean().plot.bar()
c10.pyplot(ax1.figure)
c10.write("Female students tend to score more marks in reading and writing while male students score more in maths.")
#st.plotly_chart(fig8)

c11.subheader('Based on Race/Ethnicity')
ax2=df.groupby(by='race_ethnicity')[['math_score', 'reading_score','writing_score']].mean().plot.bar()
c11.pyplot(ax2.figure)
c11.write("Students of race/ethnicity of group E tend to score more marks in all subjects than students of other groups.")
st.divider()

c12,c13=st.columns(2)
ax3=df.groupby(by='parental_level_of_education')[['math_score', 'reading_score','writing_score']].mean().plot.bar()
c12.pyplot(ax3.figure)
c12.write("Students score tend to increase directly as per their parent's education. Parents having master's degree have a significant impact on students to score more than those parents who only have a high school degree.")

ax4=df.groupby(by='lunch')[['math_score', 'reading_score','writing_score']].mean().plot.bar()
c13.pyplot(ax4.figure)
c13.write("Students scoring more who pay standard fee for their lunch can be reasoned by the fact that they might be capable of having better study environment due to their financial status. This relates directly with the parental level of education as higher educations usually leads to higher salary and financial stability.")
st.divider()

c14,c15=st.columns(2)
ax5=df.groupby(by='test_preparation_course')[['math_score', 'reading_score','writing_score']].mean().plot.bar()
c14.pyplot(ax5.figure)
c14.write("As expected, students completing test preparation course score more than students who do not.")

st.divider()