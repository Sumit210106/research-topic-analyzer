# Research Topic Analyzer

Unsupervised NLP system for analyzing research papers using clustering, topic modeling, and extractive summarization.

Built as part of an AI/ML academic project.

---

## Features

- **Text preprocessing** (spaCy + NLTK)
- **TF-IDF feature extraction**
- **Automatic K-Means clustering** (silhouette optimization)
- **Automatic DBSCAN clustering** (SVD + eps estimation)
- **LDA topic modeling**
- **Extractive summarization**
- **Streamlit UI** for visualization

---

## Dataset

**arXiv research paper dataset (Kaggle)**

[https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

> **Note:** Development uses a subset (~1000 papers) for efficiency.

---

## Tech Stack

- Python
- spaCy
- NLTK
- scikit-learn
- Streamlit
- pandas
- gensim

---

## Reproducibility

### 1. Clone repo
```bash
git clone [https://github.com/Sumit210106/research-topic-analyzer](https://github.com/Sumit210106/research-topic-analyzer)
cd research-topic-analyzer
```

### 2. Create virtual environment
```bash 
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```
### 3. Install dependencies
```bash 
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```
### 4. Run app
```bash 
streamlit run app.py
```


## Project Structure
```
    modules/
    ├── preprocessing.py
    ├── vectorization.py
    ├── clustering.py
    ├── dbscan.py
    ├── topic_modeling.py
    ├── summarization.py

    app.py
    requirements.txt
```