# 🎫 Automated IT Ticket Classification System

An NLP-based machine learning project that automates the categorization and prioritization of IT support tickets. This system helps support teams categorize issues (e.g., Hardware, Access, HR) and assess urgency (High, Medium, Low) instantly.
## [LIVE](tickettype.streamlit.app)

## 🚀 Overview

Handling high volumes of IT support tickets manually is time-consuming and prone to human error. This project utilizes **Machine Learning (Random Forest)** and **Natural Language Processing (TF-IDF)** to:
1.  **Classify Ticket Topics:** Automatically determines if a ticket is about Hardware, Access, Software, HR, etc.
2.  **Predict Priority:** Scans ticket content for urgency signals to assign Low, Medium, or High priority levels.

## 📊 Performance

* **Priority Classification Model:** 97% Accuracy
* **Topic Classification Model:** 84% Accuracy

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK/Re (Regex)
* **Models:** Random Forest Classifier
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)

## 📂 Dataset

The dataset used is the **IT Service Ticket Classification Dataset** from Kaggle.
* *Source:* [Kaggle Link](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
* *Size:* ~48,000 tickets.

## 🧠 Methodology

### 1. Data Preprocessing
* **Cleaning:** Removal of common email greetings ("Hi", "Regards"), special characters, and excessive whitespace using Regex.
* **Feature Engineering:** Created a custom `Priority` feature by scanning tickets for urgent keywords (e.g., "Critical", "Server down", "Password").

### 2. Model Training
* **Vectorization:** Converted text data into numerical vectors using `TfidfVectorizer` (Top 5000 features).
* **Algorithm:** Trained two Random Forest Classifiers with `class_weight='balanced'` to handle potential data imbalances.

### 3. Evaluation
* Evaluated using **Classification Reports** (Precision, Recall, F1-Score).
* Visualized results using **Confusion Matrices** to analyze misclassifications.
* Analyzed **Feature Importance** to see which words drove the priority decisions.

## 💻 Usage

To run this project:

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/naham6/FUTURE_ML_02.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Run the Notebook:**
    Open `MLTASK2.ipynb` in Jupyter Notebook or Google Colab.
