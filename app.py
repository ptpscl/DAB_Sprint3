import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from pyvis.network import Network
import networkx as nx
import tempfile
import re
from collections import Counter
from itertools import combinations

# ------------------ DATA LOADING ------------------
@st.cache_data

def load_data():
    df = pd.read_csv("final_dataframe.csv")  # Replace with correct path
    df.dropna(subset=['combined_description', 'combined_company'], inplace=True)
    return df

df = load_data()

# ------------------ HELPER FUNCTIONS ------------------

def clean_description(text):
    if pd.isnull(text): return ''
    text = text.lower()
    text = re.sub(r'\n|\\n', ' ', text)
    text = re.sub(r'\*\*|\*|__|_', '', text)
    text = re.sub(r'[^a-z0-9,.()\-\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(description, tech_skills, soft_skills):
    found_tech = [skill for skill in tech_skills if skill in description]
    found_soft = [skill for skill in soft_skills if skill in description]
    return found_tech, found_soft

# ------------------ SKILL LISTS ------------------
TECHNICAL_SKILLS = ["python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "aws", "azure", "gcp"]
SOFT_SKILLS = ["communication", "teamwork", "problem-solving", "leadership", "attention to detail", "adaptability"]

# ------------------ SIDEBAR ------------------
st.sidebar.title("Career Skill Explorer")
section = st.sidebar.radio("Select Research Question", [
    "RQ1: Most In-Demand Skills",
    "RQ2: Learnability of Skills",
    "RQ3: Skill Co-occurrence & Role Links",
    "RQ4: Fresh Grad-Friendly Companies",
    "Career Recommender"
])

# ------------------ RQ1 ------------------
if section == "RQ1: Most In-Demand Skills":
    st.title("RQ1: Most Common Technical and Soft Skills")
    df['cleaned'] = df['combined_description'].apply(clean_description)
    df[['tech_skills', 'soft_skills']] = df['cleaned'].apply(lambda x: pd.Series(extract_skills(x, TECHNICAL_SKILLS, SOFT_SKILLS)))

    tech_counter = Counter([s for lst in df['tech_skills'] for s in lst])
    soft_counter = Counter([s for lst in df['soft_skills'] for s in lst])

    top_tech = pd.DataFrame(tech_counter.items(), columns=['Skill', 'Count']).nlargest(10, 'Count')
    top_soft = pd.DataFrame(soft_counter.items(), columns=['Skill', 'Count']).nlargest(10, 'Count')

    st.subheader("Top 10 Technical Skills")
    st.plotly_chart(px.bar(top_tech, x='Count', y='Skill', orientation='h', title='Technical Skills'))

    st.subheader("Top 10 Soft Skills")
    st.plotly_chart(px.bar(top_soft, x='Count', y='Skill', orientation='h', title='Soft Skills'))

# ------------------ RQ2 ------------------
elif section == "RQ2: Learnability of Skills":
    st.title("RQ2: Skills - Pre-Acquired vs. On-the-Job")
    # Dummy mapping: adjust per your notebook
    pre_acquired = {"python", "sql", "r", "excel", "aws", "tableau", "power bi"}
    on_the_job = {"communication", "teamwork", "problem-solving", "adaptability", "attention to detail"}

    def classify(skill):
        skill = skill.lower()
        if skill in pre_acquired:
            return "Pre-acquired"
        elif skill in on_the_job:
            return "On-the-job"
        return "Other"

    df['learnability'] = df['tech_skills'].apply(lambda lst: [classify(skill) for skill in lst])
    skills_flat = [s for lst in df['tech_skills'] for s in lst]
    types_flat = [classify(s) for s in skills_flat]
    df_learn = pd.DataFrame({"Skill": skills_flat, "Type": types_flat})
    counts = df_learn[df_learn['Type'] != 'Other'].value_counts().reset_index(name='Count')

    st.subheader("Skill Learnability Breakdown")
    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        x='Count:Q', y='Skill:N', color='Type:N', tooltip=['Skill', 'Type', 'Count']
    ).properties(height=400), use_container_width=True)

# ------------------ RQ3 ------------------
elif section == "RQ3: Skill Co-occurrence & Role Links":
    st.title("RQ3: Skill Co-occurrence & Job Role Connections")

    st.markdown("### Co-occurrence Graph")
    # Extract skill co-occurrences
    all_skills = df['tech_skills'].dropna().tolist()
    co_counter = Counter()
    for skills in all_skills:
        for pair in combinations(set(skills), 2):
            co_counter[tuple(sorted(pair))] += 1

    G = nx.Graph()
    for (a, b), w in co_counter.items():
        if w > 2:
            G.add_edge(a, b, weight=w)

    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        st.components.v1.html(open(tmp.name, 'r').read(), height=600)

# ------------------ RQ4 ------------------
elif section == "RQ4: Fresh Grad-Friendly Companies":
    st.title("RQ4: Companies Open to Fresh Grads / Career Shifters")

    def extract_min_exp(text):
        pattern = r'(\d+)\+?\s*years?'
        match = re.search(pattern, str(text).lower())
        return int(match.group(1)) if match else 0

    df['min_exp'] = df['combined_description'].apply(extract_min_exp)
    df_filtered = df[df['min_exp'] <= 1]
    top_companies = df_filtered['combined_company'].value_counts().head(15).reset_index()
    top_companies.columns = ['Company', 'Count']

    st.subheader("Top Companies for Entry-Level Roles")
    st.plotly_chart(px.bar(top_companies, x='Count', y='Company', orientation='h'))

    st.subheader("Browse Open Roles")
    st.dataframe(df_filtered[['combined_title', 'combined_company', 'combined_description']])

# ------------------ CAREER RECOMMENDER ------------------
elif section == "Career Recommender":
    st.title("Career Recommender")
    bg = st.selectbox("What is your current background?", ["Fresh Grad", "Career Shifter", "IT Professional", "Analyst", "Other"])
    interest = st.multiselect("What are your interests?", ["Data Analysis", "Machine Learning", "Visualization", "Research", "Business Intelligence"])

    st.markdown("---")
    st.subheader("Suggested Role & Learning Path")

    if bg == "Fresh Grad" and "Data Analysis" in interest:
        st.markdown("**Recommended Role:** Data Analyst")
        st.markdown("**Start Learning:** Excel, SQL, Python, Tableau")
    elif bg == "Career Shifter" and "Business Intelligence" in interest:
        st.markdown("**Recommended Role:** BI Developer or Analyst")
        st.markdown("**Start Learning:** Power BI, SQL, Business Analytics")
    elif bg == "IT Professional" and "Machine Learning" in interest:
        st.markdown("**Recommended Role:** Data Scientist")
        st.markdown("**Start Learning:** Python, ML, Scikit-learn, TensorFlow")
    else:
        st.markdown("Use combinations of Excel, SQL, and BI tools to begin your journey. Focus on communication + technical stack.")