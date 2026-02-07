import streamlit as st
import nltk
import spacy
import secrets
import socket
import platform
import pandas as pd
import base64, random
import time, datetime
import os
import getpass
import geocoder
import io
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
import re
from pymongo import MongoClient
import plotly.express as px
from geopy.geocoders import Nominatim

# ---------- Streamlit Page Config ----------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------- Load NLTK ----------
@st.cache_resource
def load_nltk():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

load_nltk()

# ---------- Load SpaCy ----------
nlp = spacy.load("en_core_web_sm")

# ---------- MongoDB Connection ----------
@st.cache_resource
def get_mongo_client():
    return MongoClient(st.secrets["MONGO_URI"])

mongo_client = get_mongo_client()
db = mongo_client["resume_analyzer"]
user_collection = db["user_data"]
feedback_collection = db["user_feedback"]

# ---------- AI Client ----------
from google import genai
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def get_gemini_response(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return "AI Service temporarily unavailable."

# ---------- Helper Functions ----------
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Resume PDF",
                data=f,
                file_name=os.path.basename(file_path),
                mime="application/pdf"
            )
        st.info("📄 PDF preview is disabled on Streamlit Cloud. Download to view.")
    except Exception as e:
        st.error(f"Unable to load PDF: {e}")

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
    return rec_course

# ---------- Courses & Videos ----------
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# ---------- Main App ----------
def run():

    try:
        img = Image.open("Logo/logo.jpg")
        st.image(img, width="stretch")
    except:
        st.info("Logo unavailable")

    st.sidebar.markdown("# Choose Something...")
    choice = st.sidebar.selectbox("Choose among the given options:", ["User", "Admin", "Feedback", "About"])

    # ---------- USER SIDE ----------
    if choice == 'User':
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob  = st.text_input('Mobile Number*')

        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = getpass.getuser()
        os_name_ver = platform.system() + " " + platform.release()

        try:
            g = geocoder.ip('me')
            latlong = g.latlng if g.latlng else [0, 0]  # fallback
            city = g.city if hasattr(g, 'city') else ''
            state = g.state if hasattr(g, 'state') else ''
            country = g.country if hasattr(g, 'country') else ''
        except Exception:
            latlong = [0, 0]
            city = state = country = "Unknown"

        st.markdown("<h5 style='text-align: left;'>Upload Your Resume, And Get Smart Recommendations</h5>",unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])

        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)

            os.makedirs('./Uploaded_Resumes', exist_ok=True)
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            resume_text = pdf_reader(save_image_path)
            doc = nlp(resume_text)

            # ---- Extract Name ----
            lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
            blacklist = {'Pandas', 'Numpy', 'Spacy', 'Java', 'React', 'Python', 'Resume', 'CV', 'Page'}
            extracted_name = None
            for line in lines[:3]:
                line_doc = nlp(line)
                for ent in line_doc.ents:
                    if ent.label_ == "PERSON" and ent.text.strip() not in blacklist:
                        extracted_name = ent.text.strip()
                        break
                if extracted_name: break
            if not extracted_name:
                fn = pdf_file.name.split('.')[0]
                fn = re.sub(r'(?i)(resume|cv|final|updated|v\d+|20\d{2}|20\d{1})', '', fn)
                fn = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', fn)
                fn = re.sub(r'(_|-|\.)', ' ', fn)
                extracted_name = ' '.join(fn.split()).title()
            if not extracted_name or len(extracted_name) < 2:
                extracted_name = "Candidate"

            # ---- Extract Email & Phone ----
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
            phone_match = re.search(r'(\d{10}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})', resume_text)
            email = email_match.group(0) if email_match else None
            phone = phone_match.group(0) if phone_match else None

            # ---- Keywords & Skills ----
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep learning','nlp','pandas','numpy','scikit-learn','streamlit','genai','semantic analysis']
            web_keyword = ['react', 'react.js', 'next.js','node.js', 'node js', 'express', 'express.js','mongodb', 'mongo db','javascript', 'html', 'css', 'tailwind','jwt', 'rest api', 'rest apis','prisma', 'mysql', 'socket.io']
            android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
            ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
            uiux_keyword = ['adobe xd', 'figma', 'zeplin', 'balsamiq','prototyping', 'wireframes','adobe photoshop', 'illustrator','after effects', 'indesign','user research', 'user experience']
            soft_skills = ['english','communication','writing','microsoft office','leadership','customer management','social media']

            all_possible_skills = ds_keyword + web_keyword + android_keyword + ios_keyword + uiux_keyword + soft_skills
        
            found_skills = []
            for skill in all_possible_skills:
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, resume_text.lower()):
                    found_skills.append(skill)


            resume_data = {
                "name": extracted_name,
                "email": email,
                "mobile_number": phone,
                "skills": list(set(found_skills)),
                "no_of_pages": 1
            }

            st.header("**Resume Analysis**")
            st.success("Hello "+ resume_data['name'])

            st.subheader("🤖 AI Summary")
            with st.spinner('Generating AI Pitch...'):
                pitch_prompt = f"Summarize this resume into a 2-line professional pitch: {resume_text[:2500]}"
                ai_pitch = get_gemini_response(pitch_prompt)
                st.info(ai_pitch)

            st.subheader("*Your Basic info 👀*")
            try:
                st.text('Name: '+resume_data['name'])
                st.text('Email: ' + resume_data['email'])
                st.text('Contact: ' + resume_data['mobile_number'])
                st.text('Degree: '+str(resume_data['degree']))                    
                st.text('Resume pages: '+str(resume_data['no_of_pages']))

            except:
                pass
            ## Predicting Candidate Experience Level 
            ## Trying with different possibilities
            cand_level = ''
            if resume_data['no_of_pages'] < 1:
                cand_level = "NA"
                st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)

            #### if internship then intermediate level
            elif 'INTERNSHIP' in resume_text:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
            elif 'INTERNSHIPS' in resume_text:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
            elif 'Internship' in resume_text:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
            elif 'Internships' in resume_text:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)

            #### if Work Experience/Experience then Experience level
            elif 'EXPERIENCE' in resume_text:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
            elif 'WORK EXPERIENCE' in resume_text:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
            elif 'Experience' in resume_text:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
            elif 'Work Experience' in resume_text:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
            else:
                cand_level = "Fresher"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at Fresher level!!''',unsafe_allow_html=True)
            
            # ---- Skill Recommendation ----
            st.subheader("**Skills Recommendation 💡**")
            keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',
                value=resume_data['skills'],
                key = 'user_skills')

            reco_field = ''
            recommended_skills = []
            rec_course = []

            # old logic!!!
            # for i in resume_data['skills']:
            #     if i.lower() in ds_keyword:
            #         reco_field = 'Data Science'
            #         recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow','Flask','Streamlit']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Recommended skills generated from System',
            #             value=recommended_skills,key = 'rec_ds')
            #         rec_course = course_recommender(ds_course)
            #         break
            #     elif i.lower() in web_keyword:
            #         reco_field = 'Web Development'
            #         recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Recommended skills generated from System',
            #             value=recommended_skills,key = 'rec_web')
            #         rec_course = course_recommender(web_course)
            #         break
            #     elif i.lower() in android_keyword:
            #         reco_field = 'Android Development'
            #         recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Recommended skills generated from System',
            #             value=recommended_skills,key = 'rec_android')
            #         rec_course = course_recommender(android_course)
            #         break
            #     elif i.lower() in ios_keyword:
            #         reco_field = 'IOS Development'
            #         recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit','UI-Kit','AV Foundation','Auto-Layout']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Recommended skills generated from System',
            #             value=recommended_skills,key = 'rec_ios')
            #         rec_course = course_recommender(ios_course)
            #         break
            #     elif i.lower() in uiux_keyword:
            #         reco_field = 'UI-UX Development'
            #         recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Recommended skills generated from System',
            #             value=recommended_skills,key = 'rec_uiux')
            #         rec_course = course_recommender(uiux_course)
            #         break
            #     elif i.lower() in n_any:
            #         reco_field = 'NA'
            #         recommended_skills = ['No Recommendations']
            #         recommended_keywords = st_tags(label='### Recommended skills for you.',
            #             text='Currently No Recommendations',value=recommended_skills,key = 'rec_na')
            #         rec_course = ["Not Available"]
            #         break

            field_scores = {"Data Science": 0,"Web Development": 0,"Android Development": 0,"IOS Development": 0,"UI-UX Development": 0}

            for skill in resume_data['skills']:
                s = skill.lower()
                if s in ds_keyword:
                    field_scores["Data Science"] += 1
                if s in web_keyword:
                    field_scores["Web Development"] += 1   
                if s in android_keyword:
                    field_scores["Android Development"] += 1
                if s in ios_keyword:
                    field_scores["IOS Development"] += 1
                if s in uiux_keyword:
                    field_scores["UI-UX Development"] += 1

            if max(field_scores.values()) == 0:
                reco_field = "General / Undetermined"
            else:
                reco_field = max(field_scores, key=field_scores.get)

            # --------- Generate Recommendations ---------

            st.success(f"Predicted Field: {reco_field}")

            if reco_field == "Data Science":
                recommended_skills = ["Deep Learning", "Feature Engineering", "Model Deployment", "MLOps"]
                rec_course = course_recommender(ds_course)

            elif reco_field == "Web Development":
                recommended_skills = ["System Design", "Advanced Backend Architecture", "Docker", "CI/CD"]
                rec_course = course_recommender(web_course)

            elif reco_field == "Android Development":
                recommended_skills = ["Jetpack Compose", "Firebase", "MVVM Architecture"]
                rec_course = course_recommender(android_course)

            elif reco_field == "IOS Development":
                recommended_skills = ["SwiftUI", "CoreData", "App Store Deployment"]
                rec_course = course_recommender(ios_course)

            elif reco_field == "UI-UX Development":
                recommended_skills = ["Design Systems", "Interaction Design", "User Research"]
                rec_course = course_recommender(uiux_course)

            else:
                recommended_skills = ["Problem Solving", "Communication"]
                rec_course = []

    # --------- Show Recommended Skills ---------

            st.subheader("Recommended Skills to Improve")
            st_tags( label='### Recommended Skills',text='Based on your resume analysis',value=recommended_skills,key='recommended_skills')

            # ---- Resume Score ----
            st.subheader("**Resume Score 📝**")
            resume_score = 0
            # same scoring logic as before
            if 'Objective' in resume_text or 'Summary' in resume_text: resume_score+=6
            if 'Education' in resume_text or 'School' in resume_text or 'College' in resume_text: resume_score+=12
            if 'EXPERIENCE' in resume_text or 'Experience' in resume_text: resume_score+=16
            if 'INTERNSHIPS' in resume_text or 'INTERNSHIP' in resume_text: resume_score+=6
            if 'SKILLS' in resume_text or 'SKILL' in resume_text or 'Skills' in resume_text or 'Skill' in resume_text: resume_score+=7
            if 'HOBBIES' in resume_text or 'Hobbies' in resume_text: resume_score+=4
            if 'INTERESTS'in resume_text or 'Interests'in resume_text: resume_score+=5
            if 'ACHIEVEMENTS' in resume_text or 'Achievements' in resume_text: resume_score+=13
            if 'CERTIFICATIONS' in resume_text or 'Certifications' in resume_text or 'Certification' in resume_text: resume_score+=12
            if 'PROJECTS' in resume_text or 'PROJECT' in resume_text or 'Projects' in resume_text or 'Project' in resume_text: resume_score+=19

            my_bar = st.progress(0)
            for percent_complete in range(resume_score):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            st.success('** Your Resume Writing Score: ' + str(resume_score)+'**')

            # ---- Career Gap Analysis ----
            st.header("🎯 AI Career Path & Gap Analysis")
            target_job = st.selectbox("What is your target job?", ["Full Stack Developer", "Data Scientist", "DevOps Engineer", "Machine Learning Engineer", "UI/UX Designer"])
            if st.button("Analyze My Career Gap"):
                gap_prompt = f"Candidate wants to be a {target_job}. Current Skills: {resume_data['skills']}. Resume: {resume_text[:2000]}. 1. List 3 missing skills. 2. Suggest one project."
                gap_analysis = get_gemini_response(gap_prompt)
                st.markdown(gap_analysis)

            # ---- Bonus Videos ----
            st.header("**Bonus Video for Resume Writing Tips💡**")
            st.video(random.choice(resume_videos))
            st.header("**Bonus Video for Interview Tips💡**")
            st.video(random.choice(interview_videos))

            # ---- Insert into MongoDB ----
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            user_collection.insert_one({
                "act_name": act_name,
                "act_mail": act_mail,
                "act_mob": act_mob,
                "candidate_name": resume_data["name"],
                "candidate_email": resume_data["email"],
                "resume_score": resume_score,
                "total_pages": resume_data["no_of_pages"],
                "predicted_field": reco_field,
                "user_level": cand_level,
                "actual_skills": resume_data["skills"],
                "recommended_skills": recommended_skills,
                "recommended_courses": rec_course,
                "pdf_name": pdf_name,
                "timestamp": timestamp
            })

    # ---------- FEEDBACK SIDE ----------
    elif choice == 'Feedback':
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        with st.form("my_form"):
            st.write("Feedback form")
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
            comments = st.text_input('Comments')
            submitted = st.form_submit_button("Submit")
            if submitted:
                feedback_collection.insert_one({
                    "feed_name": feed_name,
                    "feed_email": feed_email,
                    "feed_score": feed_score,
                    "comments": comments,
                    "timestamp": timestamp
                })
                st.success("Thanks! Your Feedback was recorded.")
                st.balloons()

        feedbacks = list(feedback_collection.find({}, {"_id":0}))
        df_feedback = pd.DataFrame(feedbacks)
        if not df_feedback.empty:
            st.subheader("Past User Ratings")
            fig = px.pie(df_feedback, values=df_feedback['feed_score'].value_counts(), names=df_feedback['feed_score'].value_counts().index)
            st.plotly_chart(fig)

    # ---------- ABOUT PAGE ----------
    elif choice == 'About':
        st.subheader("**About The Tool - AI RESUME ANALYZER**")
        st.markdown('''
        <p align='justify'>
            A tool which parses information from a resume using NLP, finds keywords, clusters them by field, and recommends skills & courses.
        </p>
        ''', unsafe_allow_html=True)

    # ---------- ADMIN SIDE ----------
    elif choice == 'Admin':
        if "admin_logged_in" not in st.session_state:
            st.session_state.admin_logged_in = False

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            if ad_user == st.secrets["ADMIN_USER"] and ad_password == st.secrets["ADMIN_PASS"]:
                st.session_state.admin_logged_in = True
            else:
                st.error("Wrong ID & Password Provided")

        if st.session_state.admin_logged_in:
            st.success("Welcome Admin!")

            users = list(user_collection.find({}, {"_id":0}))
            df_users = pd.DataFrame(users)
            st.header("User Data")
            st.dataframe(df_users)
            st.markdown(get_csv_download_link(df_users, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)

            feedbacks = list(feedback_collection.find({}, {"_id":0}))
            df_feedback = pd.DataFrame(feedbacks)
            st.header("User Feedback Data")
            st.dataframe(df_feedback)

# ---------- Run App ----------
run()
