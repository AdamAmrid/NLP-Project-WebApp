import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import time
import re

# Page Config
st.set_page_config(page_title="Tawjih.ai", page_icon="🔮", layout="wide")

# Custom CSS
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

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

        /* Animated Hero Title */
        @keyframes shimmer {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero-title {
            background: linear-gradient(270deg, #60a5fa, #c084fc, #3b82f6);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 4rem;
            letter-spacing: -1.5px;
            margin-bottom: 8px;
            text-align: center;
            animation: shimmer 6s ease infinite;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 50px;
            font-weight: 500;
            text-align: center;
        }

        /* UNIFIED File Uploader Styling */
        /* This targets the main uploader container provided by Streamlit */
        [data-testid='stFileUploader'] {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            text-align: center;
        }

        [data-testid='stFileUploader']:hover {
            border-color: rgba(96, 165, 250, 0.4);
            box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.2);
            transform: translateY(-2px);
        }

        /* The inner dropzone section */
        [data-testid='stFileUploader'] section {
            padding: 40px 20px !important;
            background: rgba(255, 255, 255, 0.03) !important;
            border: 2px dashed rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            min-height: 200px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        [data-testid='stFileUploader'] section:hover {
            background: rgba(255, 255, 255, 0.05) !important;
            border-color: #60a5fa !important;
        }

        /* The "Browse files" button */
        [data-testid='stFileUploader'] section button {
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            margin-top: 15px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }

        [data-testid='stFileUploader'] section button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4) !important;
        }

        /* Text inside Dropzone */
        [data-testid='stFileUploader'] section span {
            font-size: 1rem !important;
            color: #cbd5e1 !important;
        }
        
        [data-testid='stFileUploader'] section small {
            font-size: 0.85rem !important;
            color: #94a3b8 !important;
            margin-bottom: 10px !important;
        }

        /* Icons */
        [data-testid='stFileUploader'] svg {
            width: 40px !important;
            height: 40px !important;
            fill: #60a5fa !important;
            margin-bottom: 10px !important;
        }

        /* Job Card Design - Expandable (Zero Gravity) */
        details.job-card {
            background: rgba(15, 23, 42, 0.6); /* Darker translucent */
            border-radius: 16px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy float */
            animation: slideIn 0.6s ease-out forwards;
            overflow: hidden;
            position: relative;
        }

        /* Hover Lift + Blue Blur Glow */
        details.job-card:hover {
            transform: translateY(-8px) scale(1.01);
            border-color: rgba(96, 165, 250, 0.5);
            box-shadow: 
                0 20px 40px -5px rgba(0, 0, 0, 0.4),
                0 0 20px rgba(59, 130, 246, 0.3); /* Blue Blur */
            background: rgba(30, 41, 59, 0.8);
        }

        details.job-card[open] {
            background: rgba(15, 23, 42, 0.9);
            border-color: rgba(96, 165, 250, 0.7);
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
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
            border-top: 1px solid rgba(255,255,255,0.05);
            margin-top: 0;
            padding-top: 20px;
            animation: fadeIn 0.4s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .score-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #34d399;
            text-shadow: 0 0 10px rgba(52, 211, 153, 0.3);
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
            transition: background 0.2s;
        }

        .meta-chip:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .meta-chip.location { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border-color: rgba(59, 130, 246, 0.2); }
        .meta-chip.type { background: rgba(139, 92, 246, 0.1); color: #a78bfa; border-color: rgba(139, 92, 246, 0.2); }

        /* Animation Keyframes */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
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
            padding-top: 3rem;
            padding-bottom: 2rem;
            max-width: 1200px;
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
st.markdown('<div class="hero-title">Tawjih.ai</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Match your profile with your dream career using advanced AI analysis.</div>', unsafe_allow_html=True)

# Spacing using columns effectively centered
col_void_l, col_main, col_void_r = st.columns([1, 10, 1])

with col_main:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        # File Uploader with integrated label in a card-like wrapper purely via CSS now
        st.markdown('<h3 style="margin-bottom: 10px; color: #f8fafc; font-size: 1.2rem;">📂 Profile Upload</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CV", type="pdf", label_visibility="collapsed")
        
        if uploaded_file:
             st.success("Analysis Complete")
             text = extract_text_from_pdf(uploaded_file)

        st.markdown("""

        <div style="margin-top: 20px; padding: 20px; background: rgba(59, 130, 246, 0.05); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.1);">
            <h4 style="margin:0; font-size: 1rem; color: #60a5fa;">💡 Pro Tip</h4>
            <p style="font-size: 0.85rem; color: #94a3b8; margin: 5px 0 0 0;">Ensure your PDF is text-selectable for best results.</p>
        </div>        """, unsafe_allow_html=True)

def is_valid_resume(text):
    """
    Validates if the text looks like a resume.
    Returns: (bool, message)
    """
    # 1. Length Check
    word_count = len(text.split())
    if word_count < 50:
        return False, "Document is too short to be a valid resume (less than 50 words)."
    if word_count > 4000:
        return False, "Document is too long to be a valid resume (more than 4000 words)."

    # 2. Email Check
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    if not re.search(email_pattern, text):
        return False, "No valid email address found in the document."

    # 3. Keyword Check
    # Case-insensitive keywords
    required_keywords = ['education', 'experience', 'skills', 'projects', 'summary', 
                         'formation', 'compétences', 'profil', 'langues']
    
    text_lower = text.lower()
    matches = sum(1 for keyword in required_keywords if keyword in text_lower)
    
    if matches < 2:
        return False, f"Document missing common resume keywords. Found {matches}, require nearly 2."

    return True, "Valid"

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
                # Validation Step
                is_valid, message = is_valid_resume(text)
                if not is_valid:
                    st.toast(f"❌ Validation Failed: {message}", icon="❌")
                    time.sleep(1) # Allow toast to appear
                    st.stop()
                
                st.success("✅ Valid Resume Detected")
                
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
                    st.balloons() # 🎉 Balloons now launch with results
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
