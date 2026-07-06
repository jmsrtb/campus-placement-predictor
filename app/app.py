import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# App Configuration
st.set_page_config(page_title="Student Employability Prediction", layout="wide")

st.title("Student Employability Prediction System")
st.markdown("---")

# Load model and columns
model = joblib.load('app/placement_model.pkl')
model_cols = joblib.load('app/model_columns.pkl')

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Student Data Input")
    st.markdown("""
    Provide the student details below to estimate the placement probability.
    - Degree Percentage: Overall academic performance (0-100).
    - Technical Skills: Self-assessed score (1-10).
    - Communication: Self-assessed score (1-10).
    - Internships: Total number of internships completed.
    """)
    
    deg_p = st.number_input("Degree Percentage (%)", 0.0, 100.0, 70.0)
    tech = st.slider("Technical Skills Score (1-10)", 1.0, 10.0, 5.0)
    comm = st.slider("Communication Score (1-10)", 1.0, 10.0, 5.0)
    intern = st.selectbox("Internships count", [0, 1, 2, 3])

if st.button("Predict Placement"):
    # Preprocessing
    input_data = pd.DataFrame({
        'degree_percentage': [deg_p], 
        'technical_skills_score': [tech],
        'communication_score': [comm], 
        'internships_count': [intern]
    })
    
    df_encoded = pd.get_dummies(input_data)
    final_input = df_encoded.reindex(columns=model_cols, fill_value=0)
    prob = model.predict_proba(final_input)[0][1]
    
    with col2:
        st.subheader("Prediction Analysis")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Placement Probability (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#007BFF"}}))
        
        # Update layout for auto-sizing to avoid warnings
        fig.update_layout(autosize=True)
        
        # Display chart without width parameters
        st.plotly_chart(fig, width='stretch')
        
        # Feedback System
        st.subheader("Improvement Suggestions")
        if prob < 0.5:
            st.warning("Assessment: Profile Needs Improvement")
            if tech < 7: st.write("Technical Skills: Focus on improving coding and technical fundamentals.")
            if comm < 7: st.write("Communication: Practice presentation and interview soft skills.")
            if intern == 0: st.write("Internships: Aim to complete at least one internship for industrial exposure.")
        else:
            st.success("Assessment: Profile is well-prepared for placement.")