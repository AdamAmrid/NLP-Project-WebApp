import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import time

# Page Config
st.set_page_config(page_title="Tawjih.ai", page_icon="🔮", layout="wide")

# Custom CSS
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        /* Root Variables */
        :root {
            --primary: #3b82f6;
            --accent: #8b5cf6;
            --glass-bg: rgba(17, 24, 39, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        /* Reset & Base */
        .stApp {
            background: 
                radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                #0f172a;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-primary);
        }

        /* Typography */
        h1, h2, h3, p, span, div {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        /* Hero Title */
        .hero-title {
            background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            letter-spacing: -1px;
            margin-bottom: 10px;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 40px;
            font-weight: 500;
        }

        /* File Uploader Container */
        .upload-container {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            backdrop-filter: blur(12px);
            transition: transform 0.3s ease;
        }
        
        .upload-container:hover {
            border-color: rgba(96, 165, 250, 0.3);
        }

        /* Streamlit File Uploader Fix */
        [data-testid='stFileUploader'] {
            width: 100%;
        }
        
        [data-testid='stFileUploader'] section {
            padding: 2px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px dashed rgba(255, 255, 255, 0.2);
            border-radius: 12px;
        }

        /* The Button itself */
        [data-testid='stFileUploader'] section button {
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            opacity: 1 !important;
        }

        [data-testid='stFileUploader'] section button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            opacity: 0.9 !important;
        }

        /* The 'Drag and drop' text and icon color */
        [data-testid='stFileUploader'] {
            color: #f8fafc !important;
        }

        /* Target the specific spans inside the uploader */
        [data-testid='stFileUploader'] section span {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        [data-testid='stFileUploader'] section small {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }

        [data-testid='stFileUploader'] svg {
            fill: #60a5fa !important;
            color: #60a5fa !important;
        }

        /* Job Card Design - Expandable */
        details.job-card {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideIn 0.5s ease-out forwards;
            overflow: hidden;
        }

        details.job-card:hover {
            transform: translateY(-4px);
            background: rgba(30, 41, 59, 0.8);
            border-color: rgba(96, 165, 250, 0.4);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
        }

        details.job-card[open] {
            background: rgba(30, 41, 59, 0.95);
            border-color: rgba(96, 165, 250, 0.6);
            transform: scale(1.01);
        }

        summary {
            list-style: none;
            padding: 24px;
            cursor: pointer;
            outline: none;
        }
        
        summary::-webkit-details-marker {
             display: none;
        }

        .job-card-content {
            padding: 0 24px 24px 24px;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.6;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 0;
            padding-top: 20px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Score Badge */
        .score-badge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 10px 16px;
            min-width: 80px;
        }

        .score-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #34d399;
        }

        .score-label {
            font-size: 0.7rem;
            color: #6ee7b7;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Metadata Chips */
        .meta-chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-right: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .meta-chip.location { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border-color: rgba(59, 130, 246, 0.2); }
        .meta-chip.type { background: rgba(139, 92, 246, 0.1); color: #a78bfa; border-color: rgba(139, 92, 246, 0.2); }

        /* Animation Keyframes */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }
        
        /* Remove default Streamlit branding spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

import os

# Helper Functions
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    try:
        # Construct absolute path to data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'preprocessed_jobs.xlsx')
        
        df = pd.read_excel(file_path)
        # Ensure processed_skills is string
        df['processed_skills'] = df['processed_skills'].astype(str).fillna("")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Main App Layout
col_header, col_void = st.columns([2, 1])
with col_header:
    st.markdown('<div class="hero-title">Tawjih.ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Match your profile with your dream career using advanced AI analysis.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.8], gap="large")

with col1:
    st.markdown("""
        <div class="upload-container">
            <h3 style="margin-top:0;">📂 Profile Upload</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Upload your CV (PDF) to begin.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CV", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
         st.success("Analysis Complete")
         text = extract_text_from_pdf(uploaded_file)
         # Debug expander removed as requested

    st.markdown("""
    <div style="margin-top: 20px; padding: 20px; background: rgba(59, 130, 246, 0.05); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.1);">
        <h4 style="margin:0; font-size: 1rem; color: #60a5fa;">💡 Pro Tip</h4>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 5px 0 0 0;">Ensure your PDF is text-selectable for best results.</p>
    </div>
    """, unsafe_allow_html=True)

import re

# Tech Keywords for Skills Analysis
TECH_KEYWORDS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular", "vue", "node.js",
    "html", "css", "sql", "mysql", "postgresql", "mongodb", "aws", "azure", "gcp", "docker", 
    "kubernetes", "git", "linux", "agile", "scrum", "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "spark", "hadoop", "tableau", 
    "power bi", "excel", "data analysis", "communication", "leadership", "project management",
    "english", "french", "spanish", "devops", "ci/cd", "terraform", "ansible", "jenkins", "statistics",
    "mathematics", "big data", "api", "rest", "graphql", "flask", "django", "spring", "dotnet"
}

def extract_skills(text):
    text = text.lower()
    found_skills = set()
    for skill in TECH_KEYWORDS:
        # Regex for robust matching (handles newlines, commas, parenthesis, etc.)
        # Special handling for skills with symbols (C++, C#, .NET)
        if skill in ['c++', 'c#']:
            # Match boundary at start, symbol at end, ensuring no word char follows
            pattern = r'\b' + re.escape(skill) + r'(?!\w)'
        elif skill.startswith('.'):
            # .NET case: specific boundary check
            pattern = re.escape(skill) + r'(?!\w)'
        else:
            # Standard words: match word boundaries on both sides
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        if re.search(pattern, text):
             found_skills.add(skill)
    return found_skills

if uploaded_file:
    with st.spinner("Finding your best matches..."):
        try:
            if not text.strip():
                st.error("Could not extract text. Please ensure the PDF contains selectable text.")
            else:
                model = load_model()
                jobs = load_data()
                
                # Extract skills from CV
                cv_skills = extract_skills(text)
                
                # Embeddings - Using processed_skills
                cv_embedding = model.encode(text, convert_to_tensor=True)
                job_embeddings = model.encode(jobs['processed_skills'].tolist(), convert_to_tensor=True)
                
                # Matching
                cosine_scores = util.cos_sim(cv_embedding, job_embeddings)[0]
                
                # Top Matches
                top_results = list(zip(range(len(cosine_scores)), cosine_scores))
                top_results.sort(key=lambda x: x[1], reverse=True)
                
                # Print header to terminal
                print("\n" + "="*50)
                print(f"Top 5 Matches for: {uploaded_file.name}")
                print("="*50)

                with col2:
                    st.markdown("### 🎯 Best Matched Opportunities")
                    for i, (idx, score) in enumerate(top_results[:5]):
                        job = jobs.iloc[idx]
                        score_percentage = round(score.item() * 100)
                        
                        # Skill Gap Analysis
                        job_text = str(job['job description']) + " " + str(job['processed_skills'])
                        job_skills = extract_skills(job_text)
                        missing_skills = list(job_skills - cv_skills)
                        
                        # Print to terminal
                        print(f"{i+1}. {job['Job title']} ({job['Source']}) - {score_percentage}% Match")
                        if missing_skills:
                            print(f"   Missing Skills: {', '.join(missing_skills)}")

                        # Generate HTML for matching skills using flat string
                        missing_skills_html = ""
                        all_missing_skills_details = ""
                        
                        if missing_skills:
                            # Summary Chips (Limit 5)
                            chips = "".join([f'<span class="meta-chip" style="background: rgba(239, 68, 68, 0.1); color: #f87171; border-color: rgba(239, 68, 68, 0.2);">⚠️ {skill}</span>' for skill in missing_skills[:5]])
                            if len(missing_skills) > 5:
                                chips += f'<span class="meta-chip" style="font-size: 0.8rem; opacity: 0.7;">+{len(missing_skills)-5} more</span>'
                            missing_skills_html = f'<div style="margin-top: 12px; margin-bottom: 8px;"><p style="font-size: 0.85rem; color: #f87171; margin-bottom: 6px; font-weight: 600;">Missing Skills:</p>{chips}</div>'

                            # Detailed View Chips (All skills)
                            all_chips = "".join([f'<span class="meta-chip" style="background: rgba(239, 68, 68, 0.1); color: #f87171; border-color: rgba(239, 68, 68, 0.2); margin-bottom: 6px;">⚠️ {skill}</span>' for skill in missing_skills])
                            all_missing_skills_details = f"""
<div style="margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 16px;">
<h4 style="color: #f87171; margin-top: 0; margin-bottom: 12px; font-size: 1.0rem;">⚠️ Detailed Skills Gap Analysis</h4>
<div style="display: flex; flex-wrap: wrap; gap: 8px;">{all_chips}</div>
</div>
"""

                        html_card = f"""
<details class="job-card" style="animation-delay: {i * 0.1}s;">
<summary>
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div style="flex: 1;">
<h3 style="margin: 0 0 4px 0; font-size: 1.4rem; color: #f8fafc;">{job['Job title']}</h3>
<p style="color: #94a3b8; font-weight: 500; margin: 0 0 16px 0; font-size: 1rem;">{job['Source']}</p>
<div style="margin-bottom: 16px;">
<span class="meta-chip type">💼 Full-time</span>
</div>
{missing_skills_html}
<p style="color: #94a3b8; font-size: 0.9rem; font-style: italic; margin-top: 10px;">▼ Click to view full details</p>
</div>
<div class="score-badge-container" style="margin-left: 20px;">
<div class="score-value">{score_percentage}%</div>
<div class="score-label">Match</div>
</div>
</div>
</summary>
<div class="job-card-content">
<h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 12px; font-size: 1.1rem;">Job Description</h4>
<p>{job['job description']}</p>
{all_missing_skills_details}
</div>
</details>
"""
                        st.markdown(html_card, unsafe_allow_html=True)
                
                print("="*50 + "\n")
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    with col2:
        st.markdown("### 🔭 Opportunity Preview")
        st.markdown("""
        <div class="job-card" style="opacity: 0.5;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h3 style="margin: 0 0 4px 0; font-size: 1.4rem; color: #f8fafc;">Senior Data Scientist</h3>
                    <p style="color: #94a3b8; font-weight: 500; margin: 0 0 16px 0; font-size: 1rem;">TechCorp Solutions</p>
                    <div style="margin-bottom: 16px;">
                        <span class="meta-chip location">📍 New York</span>
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">AI-driven matching preview. Upload your CV to unlock real opportunities tailored to you.</p>
                </div>
                <div class="score-badge-container" style="margin-left: 20px; opacity: 0.5;">
                    <div class="score-value">98%</div>
                    <div class="score-label">Match</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
