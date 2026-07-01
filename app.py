import streamlit as st
import pandas as pd
import pickle
import io

# 1. Page Configuration and Styling
st.set_page_config(page_title="Early Warning Tutor Dashboard", layout="wide")
st.title("🎓 Student Early Warning & Intervention Dashboard")
st.markdown("Upload your raw weekly class logs to instantly flag at-risk students using the Phase 2 Random Forest Model.")

# 2. Sidebar - Model Configuration & Master Key Download
st.sidebar.header("🔧 Model Settings")
threshold = st.sidebar.slider("Risk Probability Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)

# 3. File Uploader Component
uploaded_file = st.file_uploader("Choose a raw student logs Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load raw Excel data
        df = pd.read_excel(uploaded_file)
        
        # REMOVED 'MajorityVote_Difficulty' from required columns
        required_raw_cols = ['Student Name', 'Date', 'Body']
        missing_cols = [col for col in required_raw_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns in the uploaded file: {missing_cols}")
            st.stop()

        # Dynamic Generation of Custom Student IDs (Master Key Generation)
        def generate_custom_id(row, idx):
            name = str(row['Student Name']).strip()
            initials = "".join([part[0].upper() for part in name.split() if part])[:3]
            center_name = "itk"  
            return f"{idx}{center_name}{initials}"

        # Build Master Key mapping using unique Student Names
        unique_students = pd.DataFrame(df['Student Name'].unique(), columns=['Student Name']).reset_index(drop=True)
        unique_students['Student ID'] = unique_students.apply(lambda row: generate_custom_id(row, row.name + 1), axis=1)
        
        # Map the newly generated IDs back into the primary dataframe
        id_mapping = dict(zip(unique_students['Student Name'], unique_students['Student ID']))
        df['Student ID'] = df['Student Name'].map(id_mapping)

        # Allow user to download Master Key generation lookup
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Data Governance")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            unique_students.to_excel(writer, index=False, sheet_name='Master_Key_Map')
        processed_data = output.getvalue()
        
        st.sidebar.download_button(
            label="Download Master Key Map (.xlsx)",
            data=processed_data,
            file_name="student_id_master_key.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Standard Preprocessing Setup
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Student ID', 'Date'])
        
        st.success("📊 Raw file mapped and processed successfully! Running feature pipeline...")
        
        # 4. Feature Engineering Pipeline (Configured to look at raw 'Body' only)
        student_features = []
        for student_id, group in df.groupby('Student ID'):
            early_lessons = group.head(3)
            
            combined_text = " ".join(early_lessons['Body'].fillna('').astype(str))
            
            if len(early_lessons) > 1:
                date_gaps = early_lessons['Date'].diff().dt.days.dropna()
                avg_days_between_classes = date_gaps.mean()
            else:
                avg_days_between_classes = 7
                
            student_features.append({
                'Student ID': student_id,
                'Student Name': early_lessons['Student Name'].iloc[0], 
                'Combined_Text': combined_text,
                'Avg_Days_Between_Classes': round(avg_days_between_classes, 1)
            })
            
        features_df = pd.DataFrame(student_features)
        
        # 5. Load the Saved Random Forest Model
        with open('final_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # 6. Generate Predictions
        # NOTE: If your 'final_rf_model.pkl' still requires 'Avg_Early_Difficulty', this line will error out.
        X_new = features_df[['Combined_Text', 'Avg_Days_Between_Classes']]
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
            
        # Removed 'Avg_Early_Difficulty' from the display dataframe layout
        styled_df = features_df[['Student ID', 'Student Name', 'Avg_Days_Between_Classes', 'Risk Probability', 'Status']].style.map(color_status, subset=['Status'])
        
        st.dataframe(styled_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred while running the prediction framework: {e}")
else:
    st.info("💡 Please upload your raw class log workbook to begin tracking.")
