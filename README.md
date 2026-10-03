# CDS6344 Group 9 Project

## Context-Aware Aspect-Based Sentiment Analysis and Heuristic Opinion Spam-Risk Assessment for Social Networking App Reviews

This repository contains the full technical implementation for the CDS6344 Social Media Computing project by Group 9.

The project focuses on social networking app-review analysis using sentiment analysis, opinion mining, Aspect-Based Sentiment Analysis (ABSA), traditional machine learning, deep learning, transformer models, and heuristic opinion spam-risk assessment.

---

## Group Members

| Name | Student ID |
|---|---|
| Venggadanaathan A/L K. Salvam | 1231303562 |
| Tharraniah Tamilwanan | 1211111799 |

---

## Project Overview

Online app reviews contain valuable information about usability, reliability, security, communication, cost, effectiveness, and overall user experience. However, an overall review rating may not accurately represent how a user feels about each individual application feature.

This project performs **Aspect-Based Sentiment Analysis (ABSA)** on reviews from social networking applications.

The aspect-level modelling task uses the **pre-annotated aspect terms and aspect categories provided by the AWARE dataset**. The current pipeline performs **aspect-sentiment classification** and does not automatically extract new aspects or discover aspect categories.

The project also includes a **Heuristic Opinion Spam-Risk Assessment** module. Because the dataset does not contain verified spam and non-spam ground-truth labels, the system does not claim to detect confirmed spam. Instead, it identifies combinations of suspicious review characteristics and assigns a screening risk level.

High-Risk reviews are therefore treated as **heuristic screening alerts**, not confirmed spam.

---

## Dataset

The project uses the **AWARE mobile app-review dataset**.

For this project, reviews from six social networking applications were selected:

- WhatsApp Messenger
- Discord
- Facebook
- TeamSpeak 3
- Threema
- FreeTone Calling & Texting

The dataset provides review text, ratings, application names, aspect categories, aspect terms, and aspect-level sentiment labels.

### Main Processed Datasets

| Dataset | Description |
|---|---|
| `review_level_social_reviews.csv` | One row per review, used for review-level sentiment and descriptive analysis |
| `aspect_level_social_reviews.csv` | One row per annotated aspect instance, used for aspect-level sentiment classification |
| `review_level_social_reviews_with_spam_risk.csv` | Review-level dataset containing heuristic spam-risk features and screening levels |

### Final Dataset Sizes

| Dataset | Records |
|---|---:|
| Review-level dataset | 1,615 |
| Aspect-level dataset | 3,097 |
| Positive aspect instances | 1,382 |
| Negative aspect instances | 1,715 |

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
│   └── CDS6344_Group9_Notebook4_Spam_Risk_Assessment.ipynb
│
├── Diagrams and Visualizations/
│   └── Data visualisations, model evaluation charts,
│       spam-risk figures, dashboard visuals, and screenshots
│
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── data/
│   ├── assets/
│   └── model/
│       └── sentiment_roberta_finetuned/
│           # Fine-tuned model is not stored directly in GitHub
│           # because of its file size
│
├── Research Papers/
│   └── Reference materials used for the literature review,
│       where redistribution is permitted
│
└── README.md
```

---

## Notebook Summary

### Notebook 1: Data Preparation and Exploratory Data Analysis

Notebook 1 performs:

- Dataset loading and inspection
- Missing-value checking
- Duplicate checking
- Rating-based sentiment label creation
- Text cleaning and preprocessing
- Review-level dataset creation
- Aspect-level dataset creation
- Exploratory data analysis
- EDA visualisations
- Processed dataset export

**Main Outputs**

- `review_level_social_reviews.csv`
- `aspect_level_social_reviews.csv`
- Rating distribution visualisations
- Review-length visualisation
- Aspect-category visualisations
- Aspect-sentiment visualisations

---

### Notebook 2: Opinion Mining and ABSA Insights

Notebook 2 performs:

- Aspect-term frequency analysis
- Positive and negative aspect-term analysis
- Negative aspect-pattern analysis
- Opinion-word extraction
- VADER sentiment comparison
- Explicit vs implicit opinion analysis
- Rating sentiment vs aspect sentiment conflict analysis
- ABSA insight generation

**Important Findings**

| Finding | Result |
|---|---|
| Most frequent aspect term | notification |
| VADER agreement with annotated aspect sentiment | 48.37% |
| Implicit opinions | 59.06% |
| Rating-aspect sentiment conflict instances | 914 |

The relatively low VADER agreement and large proportion of implicit opinions demonstrate the difficulty of applying generic lexicon-based sentiment analysis to context-sensitive aspect opinions in app reviews.

---

### Notebook 3: Model Training and Evaluation

Notebook 3 trains and evaluates multiple aspect-level sentiment-classification models.

**Models Evaluated**

- Logistic Regression
- Linear SVM
- Random Forest
- Multinomial Naive Bayes
- BiLSTM
- DistilBERT
- Optimized RoBERTa
- Sentiment-pretrained RoBERTa with threshold tuning

#### Leakage-Aware Evaluation

The revised evaluation uses group-wise splitting by `review_id`.

All aspect instances originating from the same review remain within a single partition, preventing review-level leakage between the training, validation, and test sets.

For the neural and transformer models:

| Partition | Aspect Instances | Unique Reviews |
|---|---:|---:|
| Training | 2,156 | 1,105 |
| Validation | 354 | 184 |
| Held-out Test | 587 | 326 |

There is zero `review_id` overlap between the training, validation, and test partitions.

Traditional machine-learning models use the complete non-test pool of 2,510 aspect instances from 1,289 reviews and are evaluated on the same 587-instance held-out group-wise test set.

#### Revised Model Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 68.99% | 68.56% | 69.08% |
| Linear SVM | 69.85% | 69.25% | 69.85% |
| Random Forest | 66.78% | 65.53% | 66.46% |
| Naive Bayes | 72.40% | 69.91% | 71.13% |
| BiLSTM | 72.57% | 71.20% | 72.09% |
| DistilBERT | 78.71% | 78.48% | 78.79% |
| Optimized RoBERTa | 82.62% | 81.88% | 82.40% |
| **Sentiment RoBERTa + Threshold Tuning** | 82.45% | **81.95%** | 82.38% |

#### Final Selected Model

The final selected model is **Sentiment RoBERTa + Threshold Tuning**.

Final held-out test performance:

| Metric | Result |
|---|---:|
| Accuracy | 82.45% |
| Macro F1 | 81.95% |
| Weighted F1 | 82.38% |
| Selected threshold | 0.59 |
| Held-out test instances | 587 |

Macro F1 was defined as the primary model-selection metric.

Optimized RoBERTa achieved the highest raw Accuracy of 82.62%, while Sentiment RoBERTa + Threshold Tuning achieved the slightly higher Macro F1 of 81.95%. The sentiment-pretrained model was therefore retained as the final selected model.

#### Threshold Tuning

The Sentiment RoBERTa decision threshold was evaluated on the validation set from 0.30 to 0.70 in increments of 0.01.

The best validation threshold was **0.59**. At this threshold:

- Validation Accuracy: 84.46%
- Validation Macro F1: 84.18%

The held-out test set was not used for threshold selection.

#### Neutral Sentiment Handling

Review-level descriptive analysis contains:

- Positive
- Neutral
- Negative

However, the aspect-level modelling dataset contains only:

- Positive
- Negative

Therefore, the aspect-level classifier is binary.

The 0.30–0.70 range is a decision-threshold sweep and is not a neutral-sentiment interval.

---

### Notebook 4: Heuristic Opinion Spam-Risk Assessment

Notebook 4 performs review-level heuristic spam-risk screening.

Because the dataset does not contain verified spam/non-spam ground-truth labels, this notebook does not implement a supervised spam classifier. Instead, it uses transparent heuristic indicators to identify reviews that may deserve additional inspection.

#### Spam-Risk Features

The heuristic framework considers:

- Exact duplicate review text
- Near-duplicate review text using TF-IDF cosine similarity
- Very short reviews
- Very long reviews
- High uppercase ratio
- Repeated punctuation
- Strict promotional or external-contact patterns
- Weak commercial-term tracking
- High word repetition
- Rating deviation
- Rating-text conflict

#### Overall Heuristic Risk Distribution

| Spam-Risk Level | Count | Percentage |
|---|---:|---:|
| Low Risk | 1,094 | 67.74% |
| Medium Risk | 435 | 26.93% |
| High Risk | 86 | 5.33% |
| **Total** | **1,615** | **100.00%** |

These are screening categories, not verified spam labels.

#### Human Qualitative Validation

All 86 High-Risk heuristic flags were manually reviewed.

| Human Assessment | Count | Percentage |
|---|---:|---:|
| Clearly suspicious | 2 | 2.33% |
| Ambiguous / requires further review | 9 | 10.47% |
| Likely legitimate despite heuristic flags | 75 | 87.21% |
| **Total** | **86** | **100.00%** |

The human review shows that a High-Risk score should not be interpreted as confirmation that a review is fake, deceptive, or spam. Most High-Risk reviews appeared likely legitimate after qualitative inspection.

Therefore:

- High Risk is interpreted as a screening alert
- High Risk does not mean confirmed spam
- Reviews are not automatically removed because of their heuristic risk level
- Clearly suspicious and ambiguous cases may be prioritised for further investigation
- The manual assessments are qualitative judgments and do not constitute verified spam/not-spam ground-truth labels

#### Duplicate Screening

- Exact duplicate flagged reviews: 0
- Near-duplicate-risk reviews: 2

---

## Streamlit Application

The repository includes an interactive Streamlit application under `streamlit_app/`.

The deployed application is available at:

**https://vengga-cds6344-group9-absa-spam-detection.hf.space**

The application includes:

1. Project Overview
2. ABSA Sentiment Predictor
3. Heuristic Spam-Risk Detector
4. Project Dashboard
5. Model Summary
6. About / Documentation

### ABSA Predictor

The ABSA Predictor accepts:

- Review sentence
- Aspect category
- Aspect term

The application predicts positive or negative sentiment toward the supplied pre-annotated-style aspect context. It does not automatically extract new aspect terms or discover aspect categories.

When available, the application loads the locally stored fine-tuned Sentiment RoBERTa model. If the local model folder is unavailable, the application may use a fallback Hugging Face sentiment model.

### Fine-Tuned Model Download

The fine-tuned Sentiment RoBERTa model is not stored directly in this GitHub repository because the model files exceed normal GitHub file-size limits.

Download the model archive from Google Drive:

https://drive.google.com/file/d/1sPpQTvvYXkJtPqgw4zARuZuib4galwXv/view?usp=sharing

After downloading, extract the model into:

```text
streamlit_app/model/sentiment_roberta_finetuned/
```

The folder should contain files similar to:

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

Do **not** create an unnecessary nested folder such as:

```text
streamlit_app/model/
└── sentiment_roberta_finetuned/
    └── sentiment_roberta_finetuned/
```

The deployed model files should correspond to the final revised group-wise model used in Notebook 3.

### How to Run the Streamlit App Locally

Enter the Streamlit application directory:

```bash
cd streamlit_app
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

If the fine-tuned model is available in the correct directory, the ABSA Predictor should indicate that the local fine-tuned Sentiment RoBERTa model is being used.

### Streamlit App Requirements

Main libraries include:

```text
streamlit
pandas
numpy
matplotlib
torch
transformers
scipy
safetensors
```

---

## Final Technical Pipeline

```text
AWARE Dataset
→ Data Preparation and Cleaning
→ Exploratory Data Analysis
→ Review-Level Sentiment Analysis
→ Opinion Mining
→ Pre-Annotated Aspect-Sentiment Classification
→ Group-Wise Review-ID Splitting
→ Traditional Machine Learning
→ BiLSTM
→ Transformer Models
→ Sentiment RoBERTa Validation-Based Threshold Tuning
→ Heuristic Opinion Spam-Risk Assessment
→ Human Qualitative Validation of High-Risk Flags
→ Streamlit Demonstration
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

### Final Selected Model

| Metric | Value |
|---|---:|
| Accuracy | 82.45% |
| Macro F1 | 81.95% |
| Weighted F1 | 82.38% |
| Selected threshold | 0.59 |
| Held-out test instances | 587 |

### Highest Raw Accuracy

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Optimized RoBERTa | 82.62% | 81.88% |

Sentiment RoBERTa + Threshold Tuning remained the final selected model because Macro F1 was the primary selection metric.

### Heuristic Spam-Risk Assessment

| Metric | Result |
|---|---:|
| Low-Risk reviews | 1,094 |
| Medium-Risk reviews | 435 |
| High-Risk heuristic flags | 86 |
| High-Risk percentage | 5.33% |
| Exact duplicate flags | 0 |
| Near-duplicate-risk flags | 2 |

### Human Review of High-Risk Flags

| Assessment | Result |
|---|---:|
| Clearly suspicious | 2 |
| Ambiguous / requires further review | 9 |
| Likely legitimate despite heuristic flags | 75 |

The spam-risk framework is therefore interpreted as a screening mechanism rather than a confirmed-spam classifier.

---

## Diagrams and Visualisations

The `Diagrams and Visualizations/` folder contains visual outputs used in the report, presentation, notebooks, and Streamlit dashboard.

Examples include:

- Rating-based sentiment distribution
- Review-length distribution
- Aspect-category distribution
- Aspect-level sentiment distribution
- Top aspect-term sentiment comparison
- Positive and negative opinion-word charts
- VADER comparison visualisations
- Group-wise model comparison charts
- Sentiment RoBERTa threshold-performance plot
- Final model evaluation outputs
- Spam-risk level distribution
- Spam-feature contribution charts
- Human qualitative-validation chart
- Streamlit application screenshots

Important revised figures include:

- Accuracy and Macro F1 comparison under group-wise aspect-level evaluation
- Validation accuracy and Macro F1 across decision thresholds from 0.30 to 0.70
- Human qualitative assessment of the 86 High-Risk heuristic flags

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
- Hugging Face Spaces

---

## Important Methodological Notes

1. The aspect-level task uses pre-annotated aspect terms and categories from the AWARE dataset.
2. The current system performs aspect-sentiment classification and does not perform automatic aspect extraction.
3. Aspect-level model evaluation uses group-wise splitting by `review_id` to prevent review-level leakage.
4. The final held-out group-wise test set contains 587 aspect instances.
5. Macro F1 is used as the primary model-selection metric.
6. The final Sentiment RoBERTa decision threshold is 0.59 and was selected using validation data only.
7. Neutral sentiment is retained for review-level descriptive analysis but is not an aspect-level model class.
8. Spam-risk labels are heuristic screening levels, not verified spam labels.
9. High-Risk reviews are not automatically removed or classified as confirmed spam.
10. All 86 High-Risk reviews were qualitatively reviewed.
11. The current empirical evaluation covers six social networking applications.
12. The group-wise split is review-disjoint but not application-disjoint, so performance on completely unseen applications has not been established.
13. The Streamlit application can run without the local fine-tuned model, but in that case its ABSA Predictor may use a fallback model rather than the exact final project model.

---

## Limitations

- The study is limited to six social networking applications from the selected AWARE dataset subset.
- The evaluation prevents review-level leakage but does not perform application-disjoint testing. Therefore, the reported performance should not be interpreted as evidence of direct generalisation to completely unseen applications or other app categories.
- The aspect-level task is binary and includes positive and negative sentiment only.
- The spam-risk assessment is heuristic because verified spam/non-spam ground-truth labels are unavailable.
- The human review of the 86 High-Risk cases provides qualitative evidence about how the heuristic behaves, but it does not create a verified supervised spam benchmark.

---

## Future Work

- Evaluation on larger and more diverse app-review datasets
- Application-disjoint and unseen-app evaluation
- Multilingual review analysis
- Neutral and mixed aspect sentiment
- Automatic aspect extraction
- Joint aspect extraction and sentiment classification
- Verified spam/non-spam annotations
- Reviewer-behaviour and temporal signals
- Explainable AI methods for transformer predictions
- More advanced review-reliability models
- Improved interactive dashboard functionality

---

## Repository Access

This repository supports the CDS6344 Social Media Computing project and the associated research manuscript.

It contains the technical materials required to reproduce and inspect the analytical workflow, subject to the availability and licensing conditions of the original dataset and external model files.

### Repository Purpose

This repository supports:

- Research manuscript evidence
- Project code review
- Model evaluation
- Processed-data documentation
- Visualisation outputs
- Streamlit application demonstration
- Reproducibility of the analytical workflow

---

## Research Manuscript

The repository supports the manuscript:

**Context-Aware Aspect-Based Sentiment Analysis and Heuristic Opinion Spam-Risk Assessment for Social Networking App Reviews**

The revised evaluation includes:

- Leakage-aware group-wise splitting by `review_id`
- Expanded reproducibility details
- Revised model evaluation
- Validation-based threshold tuning
- Explicit handling of neutral sentiment
- Qualitative review of all 86 High-Risk heuristic flags
- Expanded discussion of limitations and generalisability

---

## Authors

**Group 9**
Faculty of Computing and Informatics
Multimedia University

- Venggadanaathan A/L K. Salvam - 1231303562
- Tharraniah Tamilwanan - 1211111799
