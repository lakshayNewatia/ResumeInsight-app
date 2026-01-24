import streamlit as st
import nltk, spacy, secrets, socket, platform, os, io, time, datetime, random, re, base64
import pandas as pd
import geocoder
from geopy.geocoders import Nominatim
from pymongo import MongoClient
from pdf2image import convert_from_path
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
import plotly.express as px

# ------------------ NLP SETUP ------------------
nltk.download("stopwords")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# ------------------ GEMINI ------------------
from google import genai
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def get_gemini_response(prompt):
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text.strip()
    except:
        return "AI temporarily unavailable."

# ------------------ MONGODB ------------------
@st.cache_resource
def init_mongo():
    client = MongoClient(st.secrets["MONGO_URI"])
    db = client["resume_analyzer"]
    return db

db = init_mongo()
user_data_col = db["user_data"]
feedback_col = db["user_feedback"]

# ------------------ PDF UTILS ------------------
def pdf_reader(path):
    rsrc = PDFResourceManager()
    fake = io.StringIO()
    converter = TextConverter(rsrc, fake, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrc, converter)
    with open(path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
    text = fake.getvalue()
    converter.close()
    fake.close()
    return text

def show_pdf(path):
    images = convert_from_path(path, first_page=1, last_page=1)
    if images:
        st.image(images[0], use_container_width=True)

# ------------------ INSERT FUNCTIONS ------------------
def insert_user(data):
    user_data_col.insert_one(data)

def insert_feedback(data):
    feedback_col.insert_one(data)

# ------------------ UI ------------------
st.set_page_config("AI Resume Analyzer", page_icon="📄")
st.image(Image.open("Logo/logo.jpg"))

choice = st.sidebar.selectbox("Choose", ["User", "Admin" , "Feedback", "About"])

# =====================================================
# ====================== USER =========================
# =====================================================
if choice == "User":

    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Mobile")

    pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")

    if pdf:
        os.makedirs("Uploaded_Resumes", exist_ok=True)
        path = f"Uploaded_Resumes/{pdf.name}"
        with open(path, "wb") as f:
            f.write(pdf.getbuffer())

        show_pdf(path)
        resume_text = pdf_reader(path)

        # Extract name
        extracted_name = "Candidate"
        for line in resume_text.split("\n")[:3]:
            doc = nlp(line)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    extracted_name = ent.text
                    break

        st.header("Resume Analysis")
        st.success(f"Hello {extracted_name}")

        with st.spinner("Generating AI Summary..."):
            summary = get_gemini_response(
                f"Give a 2 line professional summary:\n{resume_text[:2000]}"
            )
            st.info(summary)

        skills_db = [
            "python","java","react","django","flask","streamlit",
            "machine learning","tensorflow","sql","mongodb"
        ]

        skills = list(set([s for s in skills_db if s in resume_text.lower()]))

        st_tags("Skills", skills)

        score = len(skills) * 5
        st.progress(min(score, 100))
        st.success(f"Resume Score: {min(score,100)}")

        if st.button("Save My Analysis"):
            insert_user({
                "name": name,
                "email": email,
                "phone": phone,
                "extracted_name": extracted_name,
                "skills": skills,
                "score": score,
                "timestamp": datetime.datetime.utcnow()
            })
            st.balloons()

# =====================================================
# ==================== FEEDBACK =======================
# =====================================================
elif choice == "Feedback":
    with st.form("feedback"):
        fname = st.text_input("Name")
        femail = st.text_input("Email")
        rating = st.slider("Rating", 1, 5)
        comment = st.text_input("Comment")
        if st.form_submit_button("Submit"):
            insert_feedback({
                "name": fname,
                "email": femail,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.datetime.utcnow()
            })
            st.success("Thanks for your feedback!")
            st.balloons()

# =====================================================
# ====================== ADMIN ========================
# =====================================================
elif choice == "Admin":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login") and u == "admin" and p == "admin@resume-analyzer":

        st.header("User Data")
        users = list(user_data_col.find({}, {"_id": 0}))
        df = pd.DataFrame(users)
        st.dataframe(df)

        if not df.empty:
            fig = px.pie(df, names="score", title="Resume Scores")
            st.plotly_chart(fig)

        st.header("Feedback")
        fb = list(feedback_col.find({}, {"_id": 0}))
        st.dataframe(pd.DataFrame(fb))

# =====================================================
# ====================== ABOUT ========================
# =====================================================
else:
    st.markdown("""
    **AI Resume Analyzer**  
    Upload your resume, get AI insights, skill gaps, and career guidance.
    Built with Streamlit, Gemini AI, and MongoDB Atlas.
    """)
