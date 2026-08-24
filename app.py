import nltk
import re
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

lsvc = pickle.load(open('lsvc.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s*', ' ', txt)      
    clean_text = re.sub(r'\b(RT|cc)\b', ' ', clean_text)    
    clean_text = re.sub(r'#\S+', ' ', clean_text)       
    clean_text = re.sub(r'@\S+', ' ', clean_text)      
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',clean_text)                                                  
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)  
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()

mapping = {0: 'ACCOUNTANT',
 1: 'ADVOCATE',
 2: 'AGRICULTURE',
 3: 'APPAREL',
 4: 'ARTS',
 5: 'AUTOMOBILE',
 6: 'AVIATION',
 7: 'BANKING',
 8: 'BPO',
 9: 'BUSINESS-DEVELOPMENT',
 10: 'CHEF',
 11: 'CONSTRUCTION',
 12: 'CONSULTANT',
 13: 'DESIGNER',
 14: 'DIGITAL-MEDIA',
 15: 'ENGINEERING',
 16: 'FINANCE',
 17: 'FITNESS',
 18: 'HEALTHCARE',
 19: 'HR',
 20: 'INFORMATION-TECHNOLOGY',
 21: 'PUBLIC-RELATIONS',
 22: 'SALES',
 23: 'TEACHER'}

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume",type = ['pdf','txt'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = lsvc.predict(input_features)[0]
        st.write("Predicted Category: " ,mapping[prediction_id])

if __name__ =="__main__":
    main()
