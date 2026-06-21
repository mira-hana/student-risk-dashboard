import streamlit as st
import pandas as pd
import pickle
import io

# 1. Page Configuration and Styling
st.set_page_config(page_title="Early Warning Tutor Dashboard", layout="wide")
st.title("🎓 Student Early Warning & Intervention Dashboard")
st.markdown("Upload your weekly class logs to instantly flag at-risk students using the Phase 2 Random Forest Model.")

# 2. Sidebar - Model Configuration
st.sidebar.header("🔧 Model Settings")
# Allow tutors to dynamically tweak the threshold to see different risk levels
threshold = st.sidebar.slider("Risk Probability Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)

# 3. File Uploader Component
uploaded_file = st.file_uploader("Choose a student logs Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Student ID', 'Date'])
        
        st.success("📊 File uploaded successfully! Processing features...")
        
        # 4. Feature Engineering Pipeline
        student_features = []
        for student_id, group in df.groupby('Student ID'):
            early_lessons = group.head(3)
            combined_text = " ".join(early_lessons['Model_Ready_Text'].fillna('').astype(str))
            avg_early_difficulty = early_lessons['MajorityVote_Difficulty'].mean()
            
            if len(early_lessons) > 1:
                date_gaps = early_lessons['Date'].diff().dt.days.dropna()
                avg_days_between_classes = date_gaps.mean()
            else:
                avg_days_between_classes = 7
                
            student_features.append({
                'Student ID': student_id,
                'Combined_Text': combined_text,
                'Avg_Early_Difficulty': round(avg_early_difficulty, 2),
                'Avg_Days_Between_Classes': round(avg_days_between_classes, 1)
            })
            
        features_df = pd.DataFrame(student_features)
        
        # 5. Load the Saved Random Forest Model
        # Replace this path with where your pkl file lives locally or in your repository
        with open('final_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # 6. Generate Predictions
        X_new = features_df[['Combined_Text', 'Avg_Early_Difficulty', 'Avg_Days_Between_Classes']]
        probabilities = model.predict_proba(X_new)[:, 1]
        
        features_df['Risk Probability'] = probabilities
        features_df['Status'] = ["🚨 AT-RISK" if p >= threshold else "✅ On Track" for p in probabilities]
        
        # Sort so flagged students appear at the very top of the dashboard queue
        features_df = features_df.sort_values(by='Risk Probability', ascending=False)
        
        # 7. Display Metrics Summary Dashboard
        at_risk_count = sum(probabilities >= threshold)
        total_students = len(features_df)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Monitored Students", total_students)
        col2.metric("Flagged At-Risk Students", at_risk_count, delta=f"{at_risk_count} requires attention", delta_color="inverse")
        col3.metric("Current Decision Cutoff", f"{threshold*100}%")
        
        # 8. Interactive Data Table Display with Color Coding
        st.subheader("📋 Student Risk Assessment Queue")
        
        def color_status(val):
            color = '#ffccd5' if val == "🚨 AT-RISK" else '#d8f3dc'
            return f'background-color: {color}'
            
        styled_df = features_df[['Student ID', 'Avg_Early_Difficulty', 'Avg_Days_Between_Classes', 'Risk Probability', 'Status']].style.applymap(color_status, subset=['Status'])
        
        st.dataframe(styled_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while running the prediction framework: {e}")
else:
    st.info("💡 Please upload your 'labeled_difficulty_data.xlsx' spreadsheet to begin analysis.")