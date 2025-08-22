import streamlit as st
pg=st.navigation([st.Page("student_data_eda.py",title="Data Analysis Page"),
                 st.Page("prediction_marks.py",title="Marks Prediction")])
pg.run()