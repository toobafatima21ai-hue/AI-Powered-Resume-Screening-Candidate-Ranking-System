# 🎯 IntelliHire — AI-Powered Resume Screening, Candidate Ranking & Recruiter Copilot

> An advanced AI recruitment platform that automates resume screening, semantic candidate matching, talent discovery, and recruiter decision-making through Retrieval-Augmented Generation (RAG), FAISS vector search, transformer embeddings, and Google Gemini.

IntelliHire transforms traditional recruitment workflows by combining Natural Language Processing (NLP), Semantic Search, Vector Databases, Unsupervised Learning, and Large Language Models (LLMs) into a single end-to-end hiring assistant. Recruiters can upload resumes, rank candidates, perform semantic searches, explore talent clusters, and interact with an AI-powered recruiter chatbot that answers questions grounded in real resume content.

---
live demo link: https://huggingface.co/spaces/tooba21/AI_RESUME_SCANNER
## 🚀 Key Highlights

*  Automated Resume Parsing & Information Extraction
*  Explainable AI-Based Candidate Ranking
*  Semantic Candidate Search using FAISS
*  Transformer Embeddings with **all-MiniLM-L6-v2**
*  AI Recruiter Chatbot powered by **RAG + Gemini 1.5 Flash**
* Candidate Clustering using K-Means
*  Skill Gap Analysis & Resume Recommendations
*  AI-Generated Interview Questions
*  Local Vector Database & Persistent Candidate Search
* 📈 Interactive Streamlit Dashboard

---

## 🧠 AI & Machine Learning Architecture

IntelliHire combines multiple AI techniques into a layered recruitment intelligence pipeline:

### 1. Resume Understanding

* **spaCy (en_core_web_sm)** for Named Entity Recognition (NER)
* Regex-based extraction for emails, phone numbers, skills, education, and experience
* Section-aware resume parsing

### 2. Semantic Representation

* **Sentence Transformers**
* Model: **all-MiniLM-L6-v2**
* Generates 384-dimensional embeddings for resumes and job descriptions

### 3. Candidate Matching

* Cosine Similarity
* Skill Matching
* Experience Matching
* Explainable weighted scoring system

### 4. Vector Search Engine

* **FAISS (Facebook AI Similarity Search)**
* Resume-level semantic retrieval
* Chunk-level retrieval for RAG
* Fast nearest-neighbor search across candidate pools

### 5. Candidate Clustering

* K-Means Clustering
* Silhouette Score Optimization
* Automatic talent segmentation

### 6. Generative AI Layer

* **Google Gemini 1.5 Flash**
* Recruiter summaries
* Skill-gap analysis
* Interview question generation
* Resume improvement recommendations

---

## 🤖 RAG-Powered Recruiter Chatbot

The AI Recruiter Copilot enables recruiters to interact with their candidate pool using natural language.

Example Questions:

* "Which candidates have experience leading teams?"
* "Who has the strongest NLP background?"
* "Compare the top 3 candidates for this AI Engineer role."
* "Which candidates have both AWS and Machine Learning experience?"

### RAG Workflow

Recruiter Query
↓
MiniLM Embedding Generation
↓
FAISS Vector Retrieval
↓
Top Relevant Resume Chunks Retrieved
↓
Context Augmentation
↓
Gemini 1.5 Flash Response Generation
↓
Grounded Answer with Candidate Citations

Unlike traditional chatbots, every response is generated from retrieved resume content, making the system transparent, explainable, and less prone to hallucinations.

---

## ✨ Core Features

### Resume Processing

* Bulk PDF Upload
* Automated Candidate Profiling
* Experience Estimation
* Skill Extraction

### Job Description Analysis

* Required Skill Detection
* Preferred Skill Detection
* Experience Requirement Extraction

### Candidate Ranking

* Explainable Scoring
* Semantic Similarity Matching
* Skill-Based Evaluation
* Experience-Based Evaluation

### Semantic Search

* Natural Language Candidate Search
* FAISS-Powered Retrieval
* Context-Aware Matching

### AI Insights

* Recruiter Summaries
* Skill Gap Analysis
* Interview Question Generation
* Resume Enhancement Suggestions

### Recruiter Dashboard

* Candidate Comparison
* Talent Clustering
* Search & Filtering
* CSV Export

---

## 🏗️ System Architecture

PDF Resumes
↓
Resume Parser (spaCy + Regex)
↓
Structured Candidate Profiles
↓
SQLite Database
↓
Embedding Generation (all-MiniLM-L6-v2)
↓
FAISS Vector Index
↓
Candidate Matching & Ranking
↓
RAG Retrieval Pipeline
↓
Gemini 1.5 Flash
↓
Recruiter Dashboard & AI Copilot

---

 

## ⚙️ Explainable Candidate Scoring

```text
Final Score =
( Skill Match × 50% )
+ ( Semantic Similarity × 30% )
+ ( Experience Match × 20% )
```

| Component           | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| Skill Match         | Required & preferred skill alignment             |
| Semantic Similarity | Contextual relevance using MiniLM embeddings     |
| Experience Match    | Years of experience compared to job requirements |

Every score is fully explainable and auditable.

---

## 🛠️ Technology Stack

### Frontend

* Streamlit

### AI & NLP

* spaCy
* Sentence Transformers
* all-MiniLM-L6-v2
* Scikit-learn

### Vector Search

* FAISS

### Generative AI

* Google Gemini 1.5 Flash

### Data Processing

* Pandas
* pdfplumber

### Storage

* SQLite

### Environment Management

* python-dotenv

 

## 📊 Business Impact

* Reduces manual resume screening effort
* Improves candidate discovery through semantic search
* Enables explainable AI-driven hiring decisions
* Provides recruiters with grounded, citation-backed insights
* Supports scalable recruitment workflows

---

 
 
