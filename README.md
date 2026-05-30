# 📄 FUTURE_ML_03: Resume / Candidate Screening System

## 🌟 Project Overview
Welcome to **FUTURE_ML_03**! This project is a Machine Learning internship assignment focused on building an automated resume screening tool. The goal is to classify resumes into their best-fitting job roles using Natural Language Processing (NLP) techniques and present the results in an interactive, user-friendly Streamlit dashboard. 

## ❓ Problem Statement
Recruiters receive hundreds of resumes for a single job posting, making manual screening tedious, time-consuming, and prone to human bias. This project solves this by automatically analyzing a candidate's resume text and skills to predict their best-fit job role and generating an objective "Match Score".

## 📊 Dataset Description
The dataset used (`data/resumes.csv`) is a generated collection of realistic candidate profiles. 
It contains 300 records with the following attributes:
- `candidate_name`: The applicant's name.
- `resume_text`: A brief, realistic summary of the applicant's professional background.
- `job_role`: The target label (e.g., Data Analyst, Python Developer, ML Engineer).
- `skills`: A list of the applicant's technical skills.
- `experience_years`: Years of professional experience.

## 💻 Technologies Used
- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (TF-IDF, Logistic Regression, Naive Bayes)
- **Data Visualization:** Matplotlib, Seaborn
- **Dashboard UI:** Streamlit

## 🔄 ML Workflow
The machine learning pipeline (detailed in `notebook/resume_screening.ipynb`) follows these standard steps:
1. **Data Loading & Cleaning:** Removing missing or corrupted entries.
2. **Feature Engineering:** Combining `resume_text` and `skills` to create a rich text profile for each candidate.
3. **Text Preprocessing:** Lowercasing and removing punctuation using regular expressions.
4. **Vectorization:** Converting text data into numerical arrays using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency).
5. **Model Training:** Training both a Logistic Regression classifier and a Multinomial Naive Bayes classifier.
6. **Evaluation & Selection:** Comparing models based on accuracy and saving the best performing model.

## ✨ Features
- **Job Role Prediction:** Predicts 1 of 5 job roles based on candidate text.
- **Dynamic Skill Matching:** Highlights matched and missing skills against user-defined requirements.
- **Match Score Generation:** A custom blended metric (Model Confidence + Keyword Match).
- **Interactive UI:** A modern, sleek dark-themed dashboard.
- **Sample Data Integration:** One-click buttons to instantly load sample candidate profiles.

## 📈 Model Results
Both models were evaluated on a split test set. Due to the highly distinct and structured nature of our dataset, the models achieved excellent performance.
- **Logistic Regression Accuracy:** ~100%
- **Multinomial Naive Bayes Accuracy:** ~100%

The **Logistic Regression** model was saved as the primary classifier (`models/resume_model.pkl`) alongside the vectorizer (`models/vectorizer.pkl`).

## 🚀 Streamlit Dashboard
The interactive dashboard (`app.py`) provides an easy interface for recruiters to test candidate data without interacting with raw code.

**To run the dashboard locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Screenshots
*(Save your screenshots in the `screenshots/` folder and link them here!)*

*Example placeholder:*
<!-- ![Dashboard Overview](screenshots/dashboard.png) -->
> Please run the Streamlit app and take a screenshot to place here.

## 🔮 Future Improvements
While the current system is highly functional, future iterations could include:
- **PDF/Docx Upload:** Allow users to directly upload resume files instead of pasting text.
- **Advanced NLP:** Use deep learning (like BERT) or Large Language Models (LLMs) to capture complex context better than TF-IDF.
- **Experience Filtering:** Factor in the `experience_years` data to filter out junior vs. senior roles.
- **Broader Dataset:** Expand the training data to cover dozens of roles rather than just five.

---
*Created as part of the Machine Learning Internship Program.*
