# 📄 Resume Screening App

A machine learning-based **Resume Screening Application** built using Python and Streamlit. The application analyzes an uploaded resume and predicts the most suitable professional category using **TF-IDF vectorization** and a trained **Linear Support Vector Classifier (LinearSVC)**.

## 🚀 Features

* Upload resumes through a simple Streamlit interface
* Automatic resume text cleaning using Regular Expressions
* TF-IDF based text feature extraction
* Machine learning-based resume classification
* Supports multiple professional categories
* Displays the predicted resume category instantly
* Simple and interactive web interface using Streamlit

## 🧠 How It Works

The resume classification pipeline follows these steps:

```text
Resume
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
LinearSVC Classifier
   ↓
Predicted Job Category
```

### 1. Text Cleaning

The resume text is cleaned using Python Regular Expressions (`re`). The preprocessing removes:

* URLs
* Hashtags
* Mentions
* Special characters and punctuation
* Non-ASCII characters
* Extra whitespace

### 2. TF-IDF Vectorization

The cleaned resume text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The trained TF-IDF vectorizer is stored in:

```text
tfidf.pkl
```

### 3. Resume Classification

The TF-IDF features are passed to a trained **Linear Support Vector Classifier (LinearSVC)**.

The trained model is stored in:

```text
lsvc.pkl
```

The model predicts one of **24 resume categories**.

## 📂 Categories

The application can classify resumes into the following categories:

* Accountant
* Advocate
* Agriculture
* Apparel
* Arts
* Automobile
* Aviation
* Banking
* BPO
* Business Development
* Chef
* Construction
* Consultant
* Designer
* Digital Media
* Engineering
* Finance
* Fitness
* Healthcare
* HR
* Information Technology
* Public Relations
* Sales
* Teacher

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* NLTK
* TF-IDF Vectorization
* LinearSVC
* Pickle
* Regular Expressions

## 📁 Project Structure

```text
resume_screening/
│
├── app.py
├── resume.ipynb
├── lsvc.pkl
├── tfidf.pkl
├── archive/
├── .gitignore
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/koshijain15/resume_screening.git
```

Move into the project directory:

```bash
cd resume_screening
```

Install the required Python packages:

```bash
pip install streamlit scikit-learn nltk
```

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in your browser.

## 💻 Usage

1. Start the Streamlit application.
2. Click **Upload Resume**.
3. Select a resume file.
4. The resume text is cleaned and transformed using the trained TF-IDF vectorizer.
5. The LinearSVC model predicts the resume category.
6. The predicted category is displayed on the screen.

## 📊 Machine Learning Pipeline

The model was developed using the following workflow:

```text
Resume Dataset
      ↓
Data Preprocessing
      ↓
Text Cleaning
      ↓
Label Encoding
      ↓
Train-Test Split
      ↓
TF-IDF Vectorization
      ↓
LinearSVC Training
      ↓
Model Evaluation
      ↓
Model Serialization (.pkl)
      ↓
Streamlit Deployment
```

## 🔮 Future Improvements

* Improve PDF text extraction
* Support DOCX resumes
* Display prediction confidence
* Extract skills automatically from resumes
* Match resumes against job descriptions
* Recommend suitable job roles
* Improve the Streamlit user interface
* Deploy the application online

## 👤 Author

**Koshi Jain**

GitHub: `koshijain15`
