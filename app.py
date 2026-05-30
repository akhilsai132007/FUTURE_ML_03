import streamlit as st
import pickle
import os
import re

# Page config
st.set_page_config(page_title="Resume Screening System", layout="wide", initial_sidebar_state="expanded")

# Load models safely
@st.cache_resource
def load_models():
    model_path = 'models/resume_model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

model, vectorizer = load_models()

# 1. Title
st.title("📄 Resume / Candidate Screening System")

# 2. Sidebar with project overview
with st.sidebar:
    st.header("Project Overview")
    st.write("This machine learning dashboard screens candidate resumes and predicts the best-fit job role using TF-IDF and a trained classifier.")
    st.write("### Instructions:")
    st.write("1. Paste the candidate's resume.")
    st.write("2. Enter the target job role.")
    st.write("3. Enter required skills (comma-separated).")
    st.write("4. Click **Analyze Resume** to see the results.")
    
    st.divider()
    st.write("Developed for FUTURE_ML_03")

# 11. Add sample resume buttons
st.subheader("Quick Start: Use a Sample Resume")
col1, col2, col3 = st.columns(3)

# Session state to hold sample text
if 'sample_text' not in st.session_state:
    st.session_state['sample_text'] = ""

if col1.button("Sample Data Analyst"):
    st.session_state['sample_text'] = "I am a Data Analyst with 4 years of experience. I have strong expertise in SQL, Python, Tableau, and Data Cleaning. I have successfully delivered multiple data visualization projects."
if col2.button("Sample Python Developer"):
    st.session_state['sample_text'] = "Passionate Python Developer with a solid foundation in Python, Django, REST API, and PostgreSQL. With 6 years of professional background, I focus on writing clean code."
if col3.button("Sample ML Engineer"):
    st.session_state['sample_text'] = "Experienced Machine Learning Engineer skilled in Python, PyTorch, Deep Learning, and NLP. Over 5 years of hands-on experience building robust ML solutions and models."

# 3. Text area to paste resume text
st.subheader("1. Candidate Information")
resume_text = st.text_area("Paste Resume Text Here:", value=st.session_state['sample_text'], height=150)

# 4 & 5. Input box for required job role and required skills
st.subheader("2. Job Requirements")
col_req1, col_req2 = st.columns(2)
with col_req1:
    required_role = st.text_input("Required Job Role:", placeholder="e.g., Data Analyst")
with col_req2:
    required_skills_input = st.text_input("Required Skills (comma-separated):", placeholder="e.g., Python, SQL, Tableau")

# Helper function to extract clean skills list
def get_clean_skills(skills_str):
    if not skills_str: return []
    return [s.strip().lower() for s in skills_str.split(',') if s.strip()]

if st.button("Analyze Resume", type="primary"):
    if not resume_text:
        st.warning("Please paste a resume text to analyze.")
    elif not model or not vectorizer:
        st.error("Model or vectorizer not found. Please run train_model.py first to train and save the models.")
    else:
        st.markdown("---")
        # 6. Predict best-fit job role
        input_vector = vectorizer.transform([resume_text])
        predicted_role = model.predict(input_vector)[0]
        
        # Calculate model confidence probability
        probabilities = model.predict_proba(input_vector)[0]
        classes = model.classes_
        predicted_prob = max(probabilities)
        
        st.header("Analysis Results")
        
        # Display Predicted Role
        st.subheader("Predicted Best-Fit Role")
        if required_role and required_role.lower().strip() == predicted_role.lower():
            st.success(f"**{predicted_role}** (Matches Target Role!)")
        else:
            st.info(f"**{predicted_role}**")
        
        # Skills Analysis
        required_skills = get_clean_skills(required_skills_input)
        
        matched_skills = []
        missing_skills = []
        
        if required_skills:
            for skill in required_skills:
                if skill in resume_text.lower():
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        # 7. Show match score percentage
        skill_match_score = 0
        if required_skills:
            skill_match_score = (len(matched_skills) / len(required_skills)) * 100
        else:
            skill_match_score = 100 # If no specific skills required, assume 100% matched for skills metric
            
        model_score = predicted_prob * 100
        overall_score = (model_score + skill_match_score) / 2
        
        st.subheader(f"Overall Candidate Score: {overall_score:.1f}%")
        st.progress(int(overall_score))
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            # 8. Show matched skills
            st.write("✅ **Matched Skills**")
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"- {skill.title()}")
            else:
                st.write("None" if required_skills else "No skills specified.")
                
        with col_res2:
            # 9. Show missing skills
            st.write("❌ **Missing Skills**")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- {skill.title()}")
            else:
                st.write("None")
                
        # 10. Show candidate ranking logic
        with st.expander("Candidate Ranking Logic & Details"):
            st.write("The candidate ranking is calculated using a combination of two factors:")
            st.write(f"1. **ML Prediction Confidence ({model_score:.1f}%):** The model's probability that the resume text matches the `{predicted_role}` profile.")
            if required_skills:
                st.write(f"2. **Keyword Skill Match ({skill_match_score:.1f}%):** The percentage of required skills explicitly found in the resume text ({len(matched_skills)} out of {len(required_skills)}).")
            else:
                st.write("2. **Keyword Skill Match:** Not evaluated (no required skills provided).")
            st.write(f"**Final Score Formula:** (ML Confidence + Skill Match) / 2 = {overall_score:.1f}%")
            
            st.write("---")
            st.write("**Model Probabilities for all roles:**")
            prob_dict = dict(zip(classes, probabilities))
            for role, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {role}: {prob*100:.1f}%")
