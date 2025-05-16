import streamlit as st
import os
import shutil
import pdfplumber
import docx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# === Original Code Logic Starts ===

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

# === Streamlit UI Starts ===

st.set_page_config(layout="wide", page_title="AI Resume Screening System")
st.title("📄 AI Resume Screening and Ranking System")
st.markdown("Upload a job description and a batch of resumes to get ranked matches based on relevance.")

# File uploads
job_desc_file = st.file_uploader("Upload Job Description (.txt)", type="txt")
resume_files = st.file_uploader("Upload Resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if job_desc_file and resume_files:
    with st.spinner("🔍 Processing..."):
        # Temp directory
        temp_dir = "temp_resumes"
        os.makedirs(temp_dir, exist_ok=True)

        # Save job description
        jd_path = os.path.join(temp_dir, "job_description.txt")
        with open(jd_path, "wb") as f:
            f.write(job_desc_file.read())

        # Save resumes
        resume_paths = []
        for uploaded_file in resume_files:
            fpath = os.path.join(temp_dir, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(uploaded_file.read())
            resume_paths.append(fpath)

        # === Execute Your Code Without Changes ===
        job_description = read_txt(jd_path)
        resume_texts = []
        resume_names = []

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

        # === Dashboard ===
        st.success("✅ Ranking complete!")
        st.subheader("📌 Ranked Resumes by Relevance")

        # Table
        for rank, (filename, score) in enumerate(ranked_results, 1):
            st.write(f"**{rank}. {filename}** — Similarity Score: `{score:.4f}`")

        # Bar Chart
        st.subheader("📊 Score Distribution")
        top_k = min(20, len(ranked_results))
        top_files = [f for f, _ in ranked_results[:top_k]]
        top_scores = [float(s) for _, s in ranked_results[:top_k]]

        fig, ax = plt.subplots()
        ax.barh(top_files[::-1], top_scores[::-1], color='skyblue')
        ax.set_xlabel("Similarity Score")
        ax.set_title("Top Resume Matches")
        st.pyplot(fig)

        # Download option
        st.subheader("⬇️ Download Ranked Results")
        import pandas as pd
        df = pd.DataFrame([(rank+1, fname, float(score)) for rank, (fname, score) in enumerate(ranked_results)],
                          columns=["Rank", "Filename", "Score"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="ranked_resumes.csv", mime='text/csv')

    # Cleanup
    shutil.rmtree(temp_dir)

else:
    st.info("Please upload both a job description and at least one resume to proceed.")
