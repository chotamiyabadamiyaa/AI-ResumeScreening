import streamlit as st
import os
import shutil
import pdfplumber
import docx
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

# Text readers
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

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Resume Screening Dashboard", layout="wide")
st.title("📊 AI-Powered Resume Screening Dashboard")
st.markdown("Upload a job description and batch of resumes to explore detailed analytics and rankings.")

# Upload files
job_desc_file = st.file_uploader("📄 Upload Job Description (.txt)", type="txt")
resume_files = st.file_uploader("📑 Upload Resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if job_desc_file and resume_files:
    with st.spinner("🔍 Processing resumes..."):
        temp_dir = "temp_resumes"
        os.makedirs(temp_dir, exist_ok=True)

        jd_path = os.path.join(temp_dir, "job_description.txt")
        with open(jd_path, "wb") as f:
            f.write(job_desc_file.read())

        resume_paths, resume_texts, resume_names = [], [], []
        for uploaded_file in resume_files:
            fpath = os.path.join(temp_dir, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(uploaded_file.read())
            try:
                text = load_resume(fpath)
                if text:
                    resume_names.append(uploaded_file.name)
                    resume_paths.append(fpath)
                    resume_texts.append(text)
            except Exception as e:
                st.warning(f"Could not read {uploaded_file.name}: {e}")

        job_description = read_txt(jd_path)
        jd_embedding = model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
        cos_scores = util.cos_sim(jd_embedding, resume_embeddings)[0]

        df = pd.DataFrame({
            "Filename": resume_names,
            "Score": [float(score) for score in cos_scores],
            "Resume Text": resume_texts
        }).sort_values(by="Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        shutil.rmtree(temp_dir)

    # 🔢 Score Threshold
    st.sidebar.header("🔧 Filters")
    min_score = st.sidebar.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    search_term = st.sidebar.text_input("Search by Filename or Keyword")

    filtered_df = df[df["Score"] >= min_score]
    if search_term:
        filtered_df = filtered_df[
            filtered_df["Filename"].str.contains(search_term, case=False) |
            filtered_df["Resume Text"].str.contains(search_term, case=False)
        ]

    # 📈 KPIs
    st.subheader("📈 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", len(df))
    col2.metric("Above Threshold", len(filtered_df))
    col3.metric("Avg Score", f"{filtered_df['Score'].mean():.2f}" if not filtered_df.empty else "N/A")

    # 📊 Chart: Top Matches
    st.subheader("🏆 Top Matching Resumes")
    fig = px.bar(filtered_df.head(20), x="Score", y="Filename", orientation="h",
                 color="Score", color_continuous_scale="blues", title="Top 20 Resume Matches")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Chart: Score Distribution
    st.subheader("📊 Score Distribution")
    hist = px.histogram(filtered_df, x="Score", nbins=10, title="Score Histogram")
    st.plotly_chart(hist, use_container_width=True)

    # 🧾 Expandable Table with Resume Previews
    st.subheader("📋 Resume Table")
    for _, row in filtered_df.iterrows():
        with st.expander(f"🔍 {row['Filename']} — Score: {row['Score']:.4f}"):
            st.write(row["Resume Text"][:2000] + "..." if len(row["Resume Text"]) > 2000 else row["Resume Text"])

    # ⬇️ Download CSV
    st.download_button("Download Filtered Results CSV", data=filtered_df.to_csv(index=False),
                       file_name="filtered_ranked_resumes.csv", mime="text/csv")
else:
    st.info("Please upload a job description and multiple resumes to get started.")
