# CDS6344 Group 9 Project

## Aspect-Based Sentiment Analysis of Social Networking App Reviews Using Machine Learning, Transformer Models, and Opinion Spam Detection

This repository contains the full technical implementation for the CDS6344 Social Media Computing project by Group 9.

The project focuses on analysing social networking app reviews using sentiment analysis, opinion mining, aspect-based sentiment analysis, machine learning, deep learning, transformer models, and opinion spam-risk detection.

---

## Group Members

| Name                          | Student ID |
| ----------------------------- | ---------- |
| Venggadanaathan A/L K. Salvam | 1231303562 |
| Tharraniah Tamilwanan         | 1211111799 |

---

## Project Overview

Online app reviews contain valuable user opinions about app usability, reliability, security, cost, effectiveness, and overall user experience. However, overall star ratings may not fully represent user sentiment toward specific app features.

This project performs Aspect-Based Sentiment Analysis (ABSA) on social networking app reviews to identify what users talk about and whether their opinions toward each aspect are positive or negative.

The project also includes an opinion spam-risk detection module to identify potentially suspicious reviews based on duplicate patterns, near-duplicate patterns, rating-text conflict, rating deviation, repeated punctuation, promotional/contact signals, and other review-based indicators.

High-risk reviews are not treated as confirmed spam. They are only treated as potential spam-risk reviews that may require manual inspection.

---

## Dataset

The project uses the AWARE Social Networking App Reviews dataset.

The dataset contains reviews from social networking applications and includes review text, ratings, app names, aspect categories, aspect terms, and aspect-level sentiment labels.

Main processed datasets:

| Dataset                                          | Description                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| `review_level_social_reviews.csv`                | One row per review, used for rating-based review-level sentiment analysis |
| `aspect_level_social_reviews.csv`                | One row per aspect, used for aspect-level sentiment analysis              |
| `review_level_social_reviews_with_spam_risk.csv` | Review-level dataset with spam-risk features and spam-risk levels         |

Final dataset sizes:

| Dataset              |  Rows |
| -------------------- | ----: |
| Review-level dataset | 1,615 |
| Aspect-level dataset | 3,097 |

---

## Project Structure

```text
CDS6344_Group9_Project/
│
├── Main Dataset/
│   └── Original dataset files used for the project
│
├── data/
│   └── Processed CSV files generated from the notebooks
│
├── notebooks/
│   ├── CDS6344_Group9_Notebook1_Data_Preparation_EDA.ipynb
│   ├── CDS6344_Group9_Notebook2_OpinionMining_ABSA_Insights.ipynb
│   ├── CDS6344_Group9_Notebook3_Model_Training_Evaluation.ipynb
│   └── CDS6344_Group9_Notebook4_Opinion_Spam_Detection.ipynb
│
├── Diagrams and Visualizations/
│   └── Data visualisation outputs, model evaluation charts, dashboard visuals, and screenshots
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
├── Research Papers/
│   └── Reference materials used for literature review, if allowed to be shared
│
└── README.md
```

---

## Notebook Summary

### Notebook 1: Data Preparation and Exploratory Data Analysis

This notebook performs:

* Dataset loading and inspection
* Missing value checking
* Duplicate checking
* Rating-based sentiment label creation
* Text cleaning and preprocessing
* Review-level dataset creation
* Aspect-level dataset creation
* Exploratory data analysis
* EDA visualisations
* Cleaned dataset export

Key outputs:

* `review_level_social_reviews.csv`
* `aspect_level_social_reviews.csv`
* Rating distribution visualisations
* Review length visualisation
* Aspect category and sentiment visualisations

---

### Notebook 2: Opinion Mining and ABSA Insights

This notebook performs:

* Aspect term frequency analysis
* Positive and negative aspect term analysis
* High-risk negative aspect term detection
* Custom opinion word extraction
* VADER sentiment comparison
* Explicit vs implicit opinion analysis
* Rating sentiment vs aspect sentiment conflict analysis
* ABSA insight generation

Important findings:

| Finding                     |       Result |
| --------------------------- | -----------: |
| Most frequent aspect term   | notification |
| VADER agreement rate        |       48.37% |
| Implicit opinions           |       59.06% |
| Rating-aspect conflict rows |          914 |

---

### Notebook 3: Model Training and Evaluation

This notebook trains and evaluates multiple sentiment classification models.

Models used:

* Logistic Regression
* Linear SVM
* Random Forest
* Naive Bayes
* BiLSTM
* DistilBERT
* RoBERTa
* Sentiment RoBERTa with threshold tuning

Final best model:

| Model                                | Accuracy | Macro F1 | Weighted F1 |
| ------------------------------------ | -------: | -------: | ----------: |
| Sentiment RoBERTa + Threshold Tuning |   0.8226 |   0.8197 |      0.8221 |

The final model is used in the Streamlit app when the fine-tuned model folder is available.

---

### Notebook 4: Opinion Spam-Risk Detection

This notebook adds spam-risk detection based on opinion spam concepts and review reliability analysis.

Since the dataset does not contain verified spam and non-spam labels, this notebook uses a rule-based spam-risk detection framework instead of supervised spam classification.

Spam-risk features include:

* Exact duplicate detection
* Near-duplicate detection using TF-IDF cosine similarity
* Very short review flag
* Very long review flag
* High uppercase ratio
* Repeated punctuation
* Promotional/contact signals
* Weak commercial term tracking
* High word repetition
* Rating outlier detection
* Rating-text conflict detection

Final spam-risk results:

| Spam-Risk Level | Count | Percentage |
| --------------- | ----: | ---------: |
| Low Risk        | 1,094 |     67.74% |
| Medium Risk     |   435 |     26.93% |
| High Risk       |    86 |      5.33% |

Important note:

High-risk reviews are not confirmed spam. They are reviews that contain suspicious signals and may require manual inspection.

---

## Streamlit Application

The repository includes a Streamlit demo app in:

```text
streamlit_app/
```

The deployed Streamlit app can be accessed here:

https://vengga-cds6344-group9-absa-spam-detection.hf.space

The app includes:

1. Project overview
2. ABSA sentiment predictor
3. Opinion spam-risk detector
4. Project dashboard
5. Model summary
6. About/documentation page

The ABSA Predictor loads the local fine-tuned Sentiment RoBERTa model when available.

If the local model folder is missing, the app may use a fallback Hugging Face sentiment model.

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

If the fine-tuned model is placed correctly, the ABSA Predictor page will show that the local fine-tuned Sentiment RoBERTa model is being used.

---

## Streamlit App Requirements

The Streamlit application uses the following main libraries:

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
| --------- | ----: | ---------: |
| Positive  |   804 |     49.78% |
| Negative  |   551 |     34.12% |
| Neutral   |   260 |     16.10% |

### Aspect-Level Sentiment Distribution

| Sentiment | Count | Percentage |
| --------- | ----: | ---------: |
| Negative  | 1,715 |     55.38% |
| Positive  | 1,382 |     44.62% |

### Final Best Model

| Metric      |  Value |
| ----------- | -----: |
| Accuracy    | 82.26% |
| Macro F1    | 81.97% |
| Weighted F1 | 82.21% |

### Opinion Spam-Risk Detection

| Metric                      | Value |
| --------------------------- | ----: |
| Low-risk reviews            | 1,094 |
| Medium-risk reviews         |   435 |
| High-risk reviews           |    86 |
| High-risk percentage        | 5.33% |
| Exact duplicates            |     0 |
| Near-duplicate-risk reviews |     2 |

---

## Diagrams and Visualisations

The `Diagrams and Visualizations/` folder contains visual outputs used in the report, presentation, and Streamlit dashboard.

Examples include:

* Rating-based sentiment distribution
* Review length distribution
* Aspect category distribution
* Aspect-level sentiment distribution
* Top aspect terms sentiment comparison
* Positive and negative opinion word charts
* VADER vs dataset sentiment heatmap
* Model comparison charts
* Final best model confusion matrix
* Spam-risk level distribution
* Spam feature contribution charts
* Streamlit app screenshots

---

## Tools and Libraries

The project uses:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* NLTK
* VADER
* TensorFlow / Keras
* PyTorch
* Hugging Face Transformers
* RoBERTa
* DistilBERT
* Streamlit
* Google Colab
* GitHub
* Hugging Face Spaces

---

## Important Notes

1. The Streamlit app can run without the local fine-tuned model, but the ABSA Predictor may use a fallback Hugging Face sentiment model.
2. To use the exact final project model, the fine-tuned model folder must be placed under `streamlit_app/model/sentiment_roberta_finetuned/`.
3. Spam-risk labels are not confirmed spam labels.
4. High-risk spam reviews should be manually reviewed before making conclusions.
5. The final model score may slightly vary when rerunning transformer models due to randomness, checkpoint selection, and training environment differences.

---

## Repository Access

This repository is prepared for CDS6344 Social Media Computing project assessment. The instructor and tutor should be added as collaborators to allow repository access for marking and code review.

---

## Future Work

Future improvements can include using a larger and more diverse app review dataset to improve model generalisation. The project can also be extended by collecting verified spam and non-spam labels so that supervised spam detection models can be trained instead of relying only on rule-based spam-risk scoring.

Additional improvements include supporting multilingual app reviews, testing newer transformer or instruction-tuned language models, adding explainable AI methods to explain sentiment and spam-risk predictions, and expanding the Streamlit application with more interactive filters, real-time review input, downloadable reports, and improved cloud deployment.

---

## Repository Purpose

This repository is submitted as the Project Code Repository for the CDS6344 Social Media Computing assignment.

It supports:

* Final report
* PowerPoint presentation
* Code review
* Streamlit app demonstration
* Data visualisation screenshots
* Appendix evidence

---

## Authors

Group 9
Faculty of Computing and Informatics
Multimedia University

```text
Venggadanaathan A/L K. Salvam - 1231303562
Tharraniah Tamilwanan - 1211111799
```
