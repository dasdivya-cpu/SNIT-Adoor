import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import plotly.express as px

st.set_page_config(
    page_title=" Company Sales Dashboard ",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

df = pd.read_csv('company-sales.csv')
#st.write(df.columns[1:6])
with st.sidebar:
    add_radio = st.radio(
        "Choose a Item",
        [df.columns[1],df.columns[2],df.columns[3],df.columns[4],
        df.columns[5],df.columns[6]]
    )
st.title("XYZ shop sales Dashboard")
c1,c2,c3 = st.columns(3)
#st.write(add_radio)
#if add_radio==df.columns[1]:
placeholder = c1.empty()
c1.subheader(f"Sales details of {add_radio}")
#c1.subheader("Sales", add_radio)
c1.metric(label="Total Units Sold", value=df[add_radio].sum())
c1.bar_chart(df[add_radio],color=["#fd0"])


c2.subheader("Total units Sold for Month")
value =c2.slider("Select month", 1, len(df)-1,1)
c2.metric(label="Total units Sold for Month",value=df.loc[value,'total_units'])
a1=df.iloc[value,1:7].to_list()
#st.write(a1)
#c2.line_chart(df,x=a1)
#c2.subheader("Total units Sold for Month")
row = df[df['month_number'] == value].drop(columns=['month_number', 'total_units', 'total_profit'])

# Convert to long format for plotting
row_t = row.T.reset_index()
row_t.columns = ["Product", "Sales"]

# Line chart for that month
c2.line_chart(row_t, x="Product", y="Sales")

