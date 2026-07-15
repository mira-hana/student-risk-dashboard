import streamlit as st
import pandas as pd
import pickle
import io

st.set_page_config(page_title="Early Warning Tutor Dashboard", layout="wide")
st.title("🎓 Student Early Warning & Intervention Dashboard")
st.markdown("Upload your weekly class logs to instantly flag at-risk students using your dynamically generated optimized models.")

st.sidebar.header("🔧 Model Settings")
# Swapped st.sidebar.slider for st.sidebar.number_input to allow direct typing
threshold = st.sidebar.number_input(
    "Risk Probability Threshold", 
    min_value=0.10, 
    max_value=0.90, 
    value=0.40, 
    step=0.05
)

uploaded_file = st.file_uploader("Choose a student logs Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        # 1. Flexible Column Validation Checks
        required_raw_cols = ['Date', 'Body']
        missing_cols = [col for col in required_raw_cols if col not in df.columns]
        
        # Ensure at least one identity structural reference is present
        if 'Student Name' not in df.columns and 'Student ID' not in df.columns:
            st.error("❌ Missing identity column: Your file must contain either 'Student Name' or 'Student ID'.")
            st.stop()
            
        if missing_cols:
            st.error(f"❌ Missing required columns in the uploaded file: {missing_cols}")
            st.stop()

        # 2. Dynamic Identity Mapping and Data Governance
        if 'Student ID' not in df.columns and 'Student Name' in df.columns:
            # Generate Custom Student IDs if given raw text names
            def generate_custom_id(row, idx):
                name = str(row['Student Name']).strip()
                initials = "".join([part[0].upper() for part in name.split() if part])[:3]
                return f"{idx}itk{initials}"

            unique_students = pd.DataFrame(df['Student Name'].unique(), columns=['Student Name']).reset_index(drop=True)
            unique_students['Student ID'] = unique_students.apply(lambda row: generate_custom_id(row, row.name + 1), axis=1)
            
            id_mapping = dict(zip(unique_students['Student Name'], unique_students['Student ID']))
            df['Student ID'] = df['Student Name'].map(id_mapping)

            # Provide the downloadable master key lookup sheet in the sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("📥 Data Governance")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                unique_students.to_excel(writer, index=False, sheet_name='Master_Key_Map')
            st.sidebar.download_button(
                label="Download Master Key Map (.xlsx)",
                data=output.getvalue(),
                file_name="student_id_master_key.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            # If the file already has 'Student ID' and no 'Student Name', use 'Student ID' for display placeholders
            if 'Student Name' not in df.columns:
                df['Student Name'] = df['Student ID']
            st.sidebar.info("✅ 'Student ID' column detected directly in source data.")

        # ==============================================================================
        # STAGE 1: RUN DYNAMIC DIFFICULTY TEXT INFERENCE IN MEMORY
        # ==============================================================================
        with open('optimized_difficulty_model.pkl', 'rb') as f:
            difficulty_pipeline = pickle.load(f)
        
        text_df = pd.DataFrame({'Combined_Text': df['Body'].fillna('').astype(str)})
        df['MajorityVote_Difficulty'] = difficulty_pipeline.predict(text_df)

        # ==============================================================================
        # STAGE 2: SYNTHESIZE TIMESTAMPS AND EVALUATE RISK WINNER PIPELINE
        # ==============================================================================
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Student ID', 'Date'])
        
        student_features = []
        for student_id, group in df.groupby('Student ID'):
            early_lessons = group.head(3)
            combined_text = " ".join(early_lessons['Body'].fillna('').astype(str))
            avg_early_difficulty = early_lessons['MajorityVote_Difficulty'].mean()
            
            if len(early_lessons) > 1:
                date_gaps = early_lessons['Date'].diff().dt.days.dropna()
                avg_days_between_classes = date_gaps.mean()
            else:
                avg_days_between_classes = 7
                
            student_features.append({
                'Student ID': student_id,
                'Student Name': early_lessons['Student Name'].iloc[0], 
                'Combined_Text': combined_text,
                'Avg_Early_Difficulty': round(avg_early_difficulty, 2),
                'Avg_Days_Between_Classes': round(avg_days_between_classes, 1)
            })
            
        features_df = pd.DataFrame(student_features)
        
        with open('final_risk_model.pkl', 'rb') as f:
            risk_pipeline = pickle.load(f)
            
        X_new = features_df[['Combined_Text', 'Avg_Early_Difficulty', 'Avg_Days_Between_Classes']]
        probabilities = risk_pipeline.predict_proba(X_new)[:, 1]
        
        features_df['Risk Probability'] = probabilities
        features_df['Status'] = ["🚨 AT-RISK" if p >= threshold else "✅ On Track" for p in probabilities]
        features_df = features_df.sort_values(by='Risk Probability', ascending=False)
        
        # Display Metrics Summary Dashboards
        at_risk_count = sum(probabilities >= threshold)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Monitored Students", len(features_df))
        col2.metric("Flagged At-Risk Students", at_risk_count, delta=f"{at_risk_count} requires attention", delta_color="inverse")
        col3.metric("Current Decision Cutoff", f"{round(threshold*100, 1)}%")
        
        st.subheader("📋 Student Risk Assessment Queue")
        def color_status(val):
            return f"background-color: {'#ffccd5' if val == '🚨 AT-RISK' else '#d8f3dc'}"
            
        styled_df = features_df[['Student ID', 'Student Name', 'Avg_Early_Difficulty', 'Avg_Days_Between_Classes', 'Risk Probability', 'Status']].style.map(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while running the prediction framework: {e}")
else:
    st.info("💡 Please upload your class log workbook to begin tracking.")
