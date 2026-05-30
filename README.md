# 📄 FUTURE_ML_03: Resume / Candidate Screening System

## 🌟 Project Overview
Welcome to **FUTURE_ML_03**! This project is a comprehensive Machine Learning assignment focused on developing an automated resume screening and candidate ranking system. The goal of this project is to streamline the recruitment process by automatically classifying resumes into their best-fitting job roles and evaluating how well a candidate's skills match the specific job requirements. The underlying machine learning pipeline is exposed through an interactive, modern Streamlit dashboard.

## ✨ Features
- **Automated Job Role Prediction:** Utilizes a trained Machine Learning model to classify unstructured resume text into 1 of 5 distinct professional roles.
- **Dynamic Skill Matching:** Automatically extracts and compares skills found in the resume against recruiter-defined required skills, highlighting matches and identifying gaps.
- **Hybrid Match Scoring:** Generates an objective "Overall Candidate Score" by blending the machine learning model's predictive confidence with a deterministic keyword match percentage.
- **Interactive UI:** Features a sleek, dark-themed dashboard built with Streamlit, enabling non-technical users (like HR recruiters) to easily interact with the machine learning model.
- **Sample Data Integration:** Includes one-click quick-start buttons to instantly load sample candidate profiles for testing and demonstration.

## 💻 Technologies Used
- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF Vectorization, Logistic Regression, Multinomial Naive Bayes)
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application Framework:** Streamlit

## 📈 Model Performance
Two models were trained and evaluated on our generated candidate dataset:
- **Logistic Regression:** Achieved ~100% accuracy.
- **Multinomial Naive Bayes:** Achieved ~100% accuracy.

Due to the highly structured nature of the synthetic dataset, the models performed exceptionally well. The **Logistic Regression** model was ultimately selected and serialized (`resume_model.pkl`) alongside the TF-IDF vectorizer (`vectorizer.pkl`) for deployment in the dashboard. This demonstrates the power of TF-IDF feature extraction when combined with linear classifiers for text classification tasks.

## 🚀 Dashboard Overview
The dashboard is designed to be intuitive. Recruiters can simply paste a candidate's resume, type in the desired job role and required skills, and instantly receive a comprehensive analysis. 

To run the dashboard locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Screenshots Section

### 1. Dashboard Interface and Quick Start
![Dashboard Home](screenshots/1_dashboard_home.png)

**Description:** The landing page of the screening application featuring a modern dark UI and a sidebar containing project instructions. It includes quick-start buttons that allow users to populate the input fields with sample resumes instantly.  
**ML/Business Value:** A clean, accessible UI ensures that non-technical recruiting staff can easily leverage advanced NLP models without needing to write code or run scripts, drastically reducing friction in adoption.

### 2. Candidate Information and Job Requirements Input
![Data Input](screenshots/2_data_input.png)

**Description:** A recruiter has pasted a sample resume for a Backend Developer into the text area. Below it, they have specified the target job role ("Backend Developer") and the necessary technical skills required for the position.  
**ML/Business Value:** This demonstrates the system's flexibility. It allows human oversight to define the strict parameters (required skills) while the machine learning model analyzes the unstructured data (the resume text).

### 3. Analysis Results and Match Scoring
![Analysis Results](screenshots/3_analysis_results.png)

**Description:** The dashboard displays the results after analysis. It successfully predicts "Backend Developer" as the best-fit role and calculates an Overall Candidate Score of 77.4%. It clearly separates the matched skills from the missing ones.  
**ML/Business Value:** By highlighting both the matched and missing skills, the system provides immediate, actionable feedback to the recruiter. It moves beyond a simple "pass/fail" classification to offer a nuanced view of the candidate's strengths and weaknesses.

### 4. Transparent Ranking Logic
![Ranking Logic](screenshots/4_ranking_logic.png)

**Description:** An expandable section breaks down exactly how the 77.4% score was calculated: derived from the ML Prediction Confidence and the Keyword Skill Match. It also displays the model's confidence probabilities across all other possible job roles.  
**ML/Business Value:** Model interpretability is crucial in AI tools used for HR. By exposing the underlying probabilities and the exact mathematical formula used for scoring, the system builds trust with the user and ensures the screening process is transparent and explainable.

## 💡 Results and Insights
Through this project, several key machine learning concepts were successfully applied in a practical scenario:
- **Text Vectorization:** Learned how to convert raw, unstructured text into meaningful numerical arrays using TF-IDF, allowing algorithms to understand the importance of specific words within a document.
- **Classification Modeling:** Gained hands-on experience training, evaluating, and deploying supervised learning classifiers (Logistic Regression and Naive Bayes).
- **Model Explainability:** Realized the importance of showing *how* an AI makes decisions (via probability scores) rather than just giving a final answer, which is vital for business-facing applications.
- **Pipeline Deployment:** Bridged the gap between a Jupyter Notebook experiment and a deployed, interactive web application using Streamlit.

## 🔮 Future Improvements
To further enhance the system's capabilities, future iterations could include:
- **Document Parsing:** Implement libraries like `PyPDF2` or `docx2txt` to allow recruiters to directly upload resume files instead of copy-pasting text.
- **Advanced NLP Architectures:** Transition from TF-IDF to advanced embeddings (like Word2Vec) or Large Language Models (LLMs) to better capture context, synonyms, and semantic meaning.
- **Experience Level Filtering:** Extract and utilize the candidate's years of experience to differentiate between junior, mid-level, and senior positions automatically.
- **Real-World Dataset Training:** Fine-tune the model on a larger, messier, real-world dataset to improve its generalization and robustness to diverse resume formats.

---
*Created as part of the Machine Learning Internship Program.*
