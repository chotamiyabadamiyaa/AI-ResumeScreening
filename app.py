import streamlit as st
import os
import shutil
import pdfplumber
import docx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sentence_transformers import SentenceTransformer, util

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_resume(file_path):
    if file_path.endswith(".txt"):
        return read_txt(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".docx"):
        return read_docx(file_path)
    else:
        return None

model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(layout="wide", page_title="AI Resume Screening Dashboard")
st.title("🧠 AI-Powered Resume Screening Dashboard")
st.markdown("Upload a job description and batch of resumes to see advanced insights and rankings.")

# Uploads
job_desc_file = st.file_uploader("📄 Upload Job Description (.txt)", type="txt")
resume_files = st.file_uploader("📑 Upload Resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if job_desc_file and resume_files:
    with st.spinner("Processing resumes..."):
        temp_dir = "temp_resumes"
        os.makedirs(temp_dir, exist_ok=True)

        jd_path = os.path.join(temp_dir, "job_description.txt")
        with open(jd_path, "wb") as f:
            f.write(job_desc_file.read())

        resume_paths = []
        for uploaded_file in resume_files:
            fpath = os.path.join(temp_dir, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(uploaded_file.read())
            resume_paths.append(fpath)

        job_description = read_txt(jd_path)
        resume_texts, resume_names = [], []

        for fpath in resume_paths:
            try:
                text = load_resume(fpath)
                if text:
                    resume_names.append(os.path.basename(fpath))
                    resume_texts.append(text)
            except Exception as e:
                st.warning(f"Could not process {fpath}: {e}")

        jd_embedding = model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
        cos_scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
        ranked_results = sorted(zip(resume_names, cos_scores), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame([(rank+1, fname, float(score)) for rank, (fname, score) in enumerate(ranked_results)],
                          columns=["Rank", "Filename", "Score"])

        st.success("✅ Screening Complete")
        st.subheader("📌 Resume Rankings")
        st.dataframe(df, use_container_width=True)

        # Dashboard with Charts
        st.subheader("📊 Dashboard Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(df.head(20), x="Score", y="Filename", orientation="h", color="Score",
                          title="Top 20 Resume Matches", color_continuous_scale="Blues")
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(df, x="Score", nbins=10, title="Score Distribution", color_discrete_sequence=["indigo"])
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📈 Interactive Score Explorer")
        chart = alt.Chart(df).mark_circle(size=100).encode(
            x='Rank',
            y='Score',
            tooltip=['Rank', 'Filename', 'Score'],
            color=alt.Color('Score', scale=alt.Scale(scheme='blues'))
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Ranked CSV", data=csv, file_name="ranked_resumes.csv", mime='text/csv')

        shutil.rmtree(temp_dir)
else:
    st.info("Please upload both a job description and resume files.")
