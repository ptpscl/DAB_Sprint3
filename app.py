import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import re
from collections import Counter

# ------------------ CLEANING & SKILL EXTRACTION HELPERS ------------------
def clean_description(text):
    if pd.isnull(text): return ''
    text = text.lower()
    text = re.sub(r'\n|\\n', ' ', text)
    text = re.sub(r'\*\*|\*|__|_', '', text)
    text = re.sub(r'[^a-z0-9,.()\-\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text, tech_skills, soft_skills):
    tech_found = [skill for skill in tech_skills if skill in text]
    soft_found = [skill for skill in soft_skills if skill in text]
    return tech_found, soft_found

# ------------------ SKILL LISTS ------------------
TECHNICAL_SKILLS = ["python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "aws", "azure", "gcp"]
SOFT_SKILLS = ["communication", "teamwork", "problem-solving", "leadership", "attention to detail", "adaptability"]

# ------------------ UI ------------------
st.set_page_config(layout="wide")
st.sidebar.title("Career Skill Explorer")
section = st.sidebar.radio("Select Research Question", [
    "RQ1: Most In-Demand Skills",
    "RQ2: Learnability of Skills",
    "RQ3: Skill Co-occurrence & Role Links",
    "RQ4: Fresh Grad-Friendly Companies"
])

# ------------------ RQ1 ------------------
if section == "RQ1: Most In-Demand Skills":
    df = pd.read_csv("final_dataframe.csv")
    df.dropna(subset=['combined_description', 'combined_company'], inplace=True)
    df['cleaned'] = df['combined_description'].apply(clean_description)
    df[['tech_skills', 'soft_skills']] = df['cleaned'].apply(lambda x: pd.Series(extract_skills(x, TECHNICAL_SKILLS, SOFT_SKILLS)))

    tech_counter = Counter([s for lst in df['tech_skills'] for s in lst])
    soft_counter = Counter([s for lst in df['soft_skills'] for s in lst])

    top_tech = pd.DataFrame(tech_counter.items(), columns=['Skill', 'Count']).nlargest(10, 'Count')
    top_soft = pd.DataFrame(soft_counter.items(), columns=['Skill', 'Count']).nlargest(10, 'Count')

    st.subheader("Top 10 Technical Skills")
    st.plotly_chart(px.bar(top_tech, x='Count', y='Skill', orientation='h'))

    st.subheader("Top 10 Soft Skills")
    st.plotly_chart(px.bar(top_soft, x='Count', y='Skill', orientation='h'))

# ------------------ RQ2 ------------------
elif section == "RQ2: Learnability of Skills":
    df = pd.read_csv("final_dataframe.csv")
    df['cleaned'] = df['combined_description'].apply(clean_description)
    skills = TECHNICAL_SKILLS + SOFT_SKILLS
    df['Extracted Skills'] = df['cleaned'].apply(lambda x: [s for s in skills if s in x])

    pre = set(["sql", "python", "r", "excel", "aws", "tableau", "power bi"])
    job = set(["communication", "teamwork", "leadership", "adaptability", "problem-solving", "attention to detail"])

    def classify(skill):
        if skill in pre: return "Pre-acquired"
        elif skill in job: return "On-the-job"
        return "Other"

    all_skills = [(s, classify(s)) for lst in df['Extracted Skills'] for s in lst]
    df_class = pd.DataFrame(all_skills, columns=["Skill", "Type"])
    df_class = df_class[df_class['Type'] != "Other"]
    summary = df_class.value_counts().reset_index(name='Count')

    st.subheader("Learnability of Skills")
    st.altair_chart(alt.Chart(summary).mark_bar().encode(
        x='Count:Q', y='Skill:N', color='Type:N'
    ).properties(height=400), use_container_width=True)

# ------------------ RQ3 ------------------
elif section == "RQ3: Skill Co-occurrence & Role Links":
    st.warning("Feature under integration — coming soon with network visualizations.")

# ------------------ RQ4 ------------------
elif section == "RQ4: Fresh Grad-Friendly Companies":
    df = pd.read_csv("final_dataframe.csv")

    def extract_min_exp(text):
        match = re.search(r'(\d+)\+?\s*years?', str(text).lower())
        return int(match.group(1)) if match else 0

    df['min_exp'] = df['combined_description'].apply(extract_min_exp)
    df_filtered = df[df['min_exp'] <= 1]
    top_companies = df_filtered['combined_company'].value_counts().head(15).reset_index()
    top_companies.columns = ['Company', 'Count']

    st.subheader("Top Companies for Entry-Level Roles")
    st.plotly_chart(px.bar(top_companies, x='Count', y='Company', orientation='h'))

    st.subheader("Browse Open Roles")
    st.dataframe(df_filtered[['combined_title', 'combined_company', 'combined_description']])
