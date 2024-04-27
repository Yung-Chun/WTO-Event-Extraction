# Project: WTO Event Extraction using BERTopic framework

## Overview

This repository employs BERTopic framework to apply topic modeling to the publications on the World Trade Organization (WTO) website. By leveraging `Sentence-BERT`, `UMAP`, `HDBSCAN`, and `c-TF-IDF`, we are able to group similar documents as an event.

## Objective

In this project, I employ BERTopic for event extraction. BERTopic (Grootendorst 2022) is a topic modeling framework that diverges from traditional Bayesian network approaches by utilizing a density-based clustering algorithm. The framework entails several crucial steps: firstly, leveraging Sentence-BERT (SBERT), a BERT-based sentence transformer, to generate embeddings for subsequent clustering. Secondly, after dimensionality reduction using UMAP, documents are clustered using HDBSCAN. Thirdly, all documents within a cluster are treated as a single document for representation, employing tokenization and class-based TF-IDF (c-TF-IDF) to unify the representation for a group of clustered documents.

## Methodology 

For data cleansing, see `WTO-data-cleansing.ipynb`. 
For preprocessing and topic modeling, see `WTO-BERTopic-topic-modeling.ipynb`

### Data Cleansing

The data is collected using the [WTO-News-and-Events-Archive-Crawler](https://github.com/Yung-Chun/WTO-News-and-Events-Archive-Crawler) repository. The crawler extracted 10,110 unique URLs, including articles, and audio and video content. At this stage, I have focused solely on textual data, processing both abstracts and full content, and applied language detection using spaCy. After filtering for English articles with standardized date formats, 8,296 articles remained, spanning from 1991 to 2023. 

### Preprocessing

The preprocessing steps include:
1. Removing punctuation
2. Converting to lowercase
3. Lemmatization
4. Removing stopwords and irrelevant terms. (Note: Commonly used words that do not contribute to the semantic meaning, such as "the", "is", and "in", are removed alongside specific terms that might skew the representation of the text's content. These include words like "download", "pdf", "word", "share", and any URLs, as they are often not relevant to the core analysis.)
5. Eliminating Digits and Extra Spaces. (Note: All numbers are removed to focus purely on textual data. Additionally, multiple spaces are reduced to a single space to maintain text consistency and readability.)

### Embeddings

There are several models in SBERT (Reimers 2019). With the consideration of both performance and time efficiency, this project uses `all-MiniLM-L6-v2` as the transformer model. `all-MiniLM-L6-v2` is one of the latest models fine-tuned with multiple datasets, including over 1 billion sentence pairs.

### Topic Modeling

There are two hyperparameters to optimize BERTopic's performance: the minimum topic size (`min_topic_size`) and the similarity threshold for zero-shot classification (`zeroshot_min_similarity`). After conducting an experimentation varying min_topic_size from 2 to 10 and adjusting zeroshot_min_similarity in 0.05 increments, ranging from 0.7 to 0.9, `min_topic_size=4` and `zeroshot_min_similarity=0.7` was selected to maximize the number of discernible topics while maintaining interpretability. In addition, the model allows unclassified topics, avoiding the ambiguity caused by forcibly merging scattered documents to coherent topics.

### Topic Merging

To refine the topic model further, you can use Hierarchical Topic Modeling (HTM) to identify similar topics and then leverage domain-specific knowledge to integrate topics with either identical or highly similar thematic representations. 

### Results and Deliverables

For evaluation metrics, see folder `evaluation`. For results of topic modeling, including topic-level and document-level, see folder `results`. If you want to inquiry other information or apply the model directly, `WTO_BERTopic_trained.pkl` is the original model and `WTO_BERTopic_merged.pkl` is the one after merging.

### Requirements 

bertopic==0.16.0; 
gensim==4.3.2; 
kneed==0.8.5; 
numpy==1.22.4; 
pandas==2.2.1; 
plotly==5.19.0; 
requests==2.31.0; 
scikit-learn==1.4.1.post1; 
scipy==1.11.4; 
spacy==3.7.4; 
tqdm==4.66.2; 

Also see `requirements.txt`.




