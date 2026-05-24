[README.md](https://github.com/user-attachments/files/28190475/README.md)
# CDS6344 Group 9 Project  
## Aspect-Based Sentiment Analysis of Social Networking App Reviews Using Machine Learning, Transformer Models, and Opinion Spam Detection

This repository contains the full technical implementation for the CDS6344 Social Media Computing project by Group 9.

The project focuses on analysing social networking app reviews using sentiment analysis, opinion mining, aspect-based sentiment analysis, machine learning, deep learning, transformer models, and opinion spam-risk detection.

---

## Group Members

| Name | Student ID |
|---|---|
| Venggadanaathan | 1231303562 |
| Tharraniah Tamilwanan | 1211111799 |

---

## Project Overview

Online app reviews contain valuable user opinions about app usability, reliability, security, cost, effectiveness, and overall experience. However, overall star ratings may not fully represent user sentiment toward specific app features.

This project performs **Aspect-Based Sentiment Analysis (ABSA)** on social networking app reviews to identify what users talk about and whether their opinions toward each aspect are positive or negative.

The project also includes an **opinion spam-risk detection module** to identify potentially suspicious reviews based on duplicate patterns, rating-text conflict, rating deviation, repeated punctuation, promotional signals, and other review-centric indicators.

---

## Dataset

The project uses the **AWARE Social Networking App Reviews dataset**.

The dataset contains reviews from social networking applications and includes review text, ratings, app names, aspect categories, aspect terms, and aspect-level sentiment labels.

Main processed datasets:

| Dataset | Description |
|---|---|
| `review_level_social_reviews.csv` | One row per review, used for rating-based document-level sentiment |
| `aspect_level_social_reviews.csv` | One row per aspect, used for aspect-level sentiment analysis |
| `review_level_social_reviews_with_spam_risk.csv` | Review-level dataset with spam-risk features and labels |

Final dataset sizes:

| Dataset | Rows |
|---|---:|
| Review-level dataset | 1,615 |
| Aspect-level dataset | 3,097 |

---

## Project Structure

```text
CDS6344_Group9_Project/
│
├── notebooks/
│   ├── CDS6344_Group9_Notebook1_Data_Preparation_EDA.ipynb
│   ├── CDS6344_Group9_Notebook2_OpinionMining_ABSA_Insights.ipynb
│   ├── CDS6344_Group9_Notebook3_Model_Training_Evaluation.ipynb
│   └── CDS6344_Group9_Notebook4_Opinion_Spam_Detection.ipynb
│
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── data/
│   ├── assets/
│   └── model/
│       └── sentiment_roberta_finetuned/   # Not included in GitHub due to file size
│
└── README.md
```

---

## Notebook Summary

### Notebook 1: Data Preparation and Exploratory Data Analysis

This notebook performs:

- Dataset loading and inspection
- Missing value checking
- Duplicate checking
- Rating-based sentiment label creation
- Text cleaning and preprocessing
- Review-level dataset creation
- Aspect-level dataset creation
- Exploratory data analysis
- EDA visualizations
- Cleaned dataset export

Key outputs:

- `review_level_social_reviews.csv`
- `aspect_level_social_reviews.csv`
- Rating distribution visualizations
- Aspect category and sentiment visualizations

---

### Notebook 2: Opinion Mining and ABSA Insights

This notebook performs:

- Aspect term frequency analysis
- Positive and negative aspect term analysis
- High-risk negative aspect term detection
- Custom opinion word extraction
- VADER sentiment comparison
- Explicit vs implicit opinion analysis
- Rating sentiment vs aspect sentiment conflict analysis
- ABSA insight generation

Important findings:

| Finding | Result |
|---|---:|
| Most frequent aspect term | notification |
| VADER agreement rate | 48.37% |
| Implicit opinions | 59.06% |
| Rating-aspect conflict rows | 914 |

---

### Notebook 3: Model Training and Evaluation

This notebook trains and evaluates multiple sentiment classification models.

Models used:

- Logistic Regression
- Linear SVM
- Random Forest
- Naive Bayes
- BiLSTM
- DistilBERT
- RoBERTa
- Sentiment RoBERTa with threshold tuning

Final best model:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Sentiment RoBERTa + Threshold Tuning | 0.8226 | 0.8197 | 0.8221 |

The final model is used in the local Streamlit app when the fine-tuned model folder is available.

---

### Notebook 4: Opinion Spam-Risk Detection

This notebook adds spam-risk detection based on the lecturer’s requirement and the opinion spam concepts discussed in the course.

Since the dataset does not contain verified spam/not-spam labels, this notebook uses a **rule-based spam-risk detection framework** instead of supervised spam classification.

Spam-risk features include:

- Exact duplicate detection
- Near-duplicate detection using TF-IDF cosine similarity
- Very short review flag
- Very long review flag
- High uppercase ratio
- Repeated punctuation
- Strict promotional/contact signals
- Weak commercial term tracking
- High word repetition
- Rating outlier detection
- Rating-text conflict detection

Final spam-risk results:

| Spam-Risk Level | Count | Percentage |
|---|---:|---:|
| Low Risk | 1,094 | 67.74% |
| Medium Risk | 435 | 26.93% |
| High Risk | 86 | 5.33% |

Important note:

High-risk reviews are **not confirmed spam**. They are reviews that require manual inspection.

---

## Streamlit App

The repository includes a Streamlit demo app in:

```text
streamlit_app/
```

The app includes:

1. Project overview
2. ABSA sentiment predictor
3. Opinion spam-risk detector
4. Project dashboard
5. Model summary
6. About/documentation page

The ABSA Predictor loads the local fine-tuned Sentiment RoBERTa model when available.

If the local model folder is missing, the app falls back to a Hugging Face sentiment-pretrained RoBERTa model.

---

## Fine-Tuned Model Download

The fine-tuned Sentiment RoBERTa model is not included in this GitHub repository because the model file is too large for normal GitHub storage.

Download the model from Google Drive:

```text
https://drive.google.com/file/d/1sPpQTvvYXkJtPqgw4zARuZuib4galwXv/view?usp=sharing
```

After downloading, unzip the model and place it in:

```text
streamlit_app/model/sentiment_roberta_finetuned/
```

The final folder should look like this:

```text
streamlit_app/
└── model/
    └── sentiment_roberta_finetuned/
        ├── config.json
        ├── model.safetensors or pytorch_model.bin
        ├── tokenizer_config.json
        ├── vocab.json
        ├── merges.txt
        ├── special_tokens_map.json
        └── streamlit_model_config.json
```

Do not place the folder like this:

```text
streamlit_app/model/sentiment_roberta_finetuned/sentiment_roberta_finetuned/
```

---

## How to Run the Streamlit App Locally

Go into the Streamlit app folder:

```bash
cd streamlit_app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

If the fine-tuned model is placed correctly, the ABSA Predictor page will show:

```text
Using local fine-tuned Sentiment RoBERTa model
```

---

## Streamlit App Requirements

```text
streamlit
pandas
matplotlib
torch
transformers
scipy
safetensors
```

---

## Final Technical Pipeline

```text
Dataset
→ Data Cleaning
→ Exploratory Data Analysis
→ Rating-Based Sentiment Labelling
→ Opinion Mining
→ Aspect-Based Sentiment Analysis
→ Traditional Machine Learning
→ BiLSTM Deep Learning
→ Transformer Models
→ Sentiment RoBERTa Threshold Tuning
→ Opinion Spam-Risk Detection
→ Streamlit App Demo
```

---

## Key Results

### Review-Level Sentiment Distribution

| Sentiment | Count | Percentage |
|---|---:|---:|
| Positive | 804 | 49.78% |
| Negative | 551 | 34.12% |
| Neutral | 260 | 16.10% |

### Aspect-Level Sentiment Distribution

| Sentiment | Count | Percentage |
|---|---:|---:|
| Negative | 1,715 | 55.38% |
| Positive | 1,382 | 44.62% |

### Final Best Model

| Metric | Value |
|---|---:|
| Accuracy | 82.26% |
| Macro F1 | 81.97% |
| Weighted F1 | 82.21% |

### Opinion Spam-Risk Detection

| Metric | Value |
|---|---:|
| High-risk reviews | 86 |
| High-risk percentage | 5.33% |
| Exact duplicates | 0 |
| Near-duplicate-risk reviews | 2 |

---

## Tools and Libraries

The project uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- VADER
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- RoBERTa
- DistilBERT
- Streamlit
- Google Colab
- GitHub

---

## Important Notes

1. The Streamlit app can run without the local fine-tuned model, but the ABSA Predictor will use a fallback Hugging Face sentiment model.
2. To use the exact final project model, the fine-tuned model folder must be placed under `streamlit_app/model/sentiment_roberta_finetuned/`.
3. Spam-risk labels are not confirmed spam labels.
4. High-risk spam reviews should be manually reviewed before making conclusions.
5. The final model score slightly changed after rerunning because transformer training may vary slightly due to randomness and checkpoint selection.

---

## Repository Purpose

This repository is submitted as the **Project Code Repository** for the CDS6344 Social Media Computing assignment.

It supports:

- Final report
- PowerPoint presentation
- Code review
- Streamlit app demonstration
- Appendix evidence

---

## Authors

Group 9  
Faculty of Computing and Informatics  
Multimedia University

```text
Venggadanaathan - 1231303562
Tharraniah Tamilwanan - 1211111799
```
