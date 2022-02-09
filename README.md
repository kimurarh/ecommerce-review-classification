# E-commerce Review Classification

## Project

In this project, I'll be exploring a Brazilian e-commerce public dataset of orders made at Olist Store available on [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce). The dataset has information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil.

In my other project ([olist-public](https://github.com/kimurarh/olist-public)) I have conducted a more detailed exploratory analysis on all datasets made publicly available by Olist. Here, as we'll only work with textual data on order reviews, I will only conduct a brief exploratory analysis on the `Order Reviews` dataset.

The customers written reviews will be pre-processed using Natural Language Processing (NLP) techniques and used as inputs to Machine Learning models.

## Development Steps

1. Exploratory Data Analysis;
2. Text Pre-processing using NLP techniques;
3. Explore Vectorization techniques (Bag of Words, TF-IDF, Word2Vec);
4. Implement text classification models (positive or negative reviews).

## Technologies
The following technologies were used in this project:

* Python
* Numpy
* Matplotlib
* Seaborn
* NLTK
* Gensim (Word2Vec)
* Scikit-learn
* XGBoost

## File Organization

    .
    ├── assets/                                             # Project assets                                           
    ├── data/                                               # E-commerce dataset                        
    ├── 0-Exploratory-Data-Analysis.ipynb
    ├── 1-Text-Preprocessing.ipynb
    ├── 2-Vectorization-Methods-and-Classification.ipynb
    ├── utils.py                                            # Utility functions    
    └── README.md                                           # Documentation

## How to Use
Create the conda environment from the environment.yml file available in this repository:
```
conda env create --name <ENVIRONMENT_NAME> -f environment.yml
```
Activate the conda environment:
```
conda activate <ENVIRONMENT_NAME>
```
Launch jupyter notebook:
```
jupyter notebook
```
- The `jupyter` application will run in your default browser;
- Select the `notebook` (`.ipynb` file) you want to execute.
