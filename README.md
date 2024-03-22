# Movie Review Sentiment Classification Assignment Report

## Introduction

This document details the execution and insights from a sentiment classification project, part of an advanced machine learning course curriculum. The project's core objective was to construct a model capable of accurately classifying IMDb movie reviews into positive or negative sentiments. This endeavor aimed to demonstrate the practical application of logistic regression techniques in conjunction with dense word vectors, particularly within the domain of Natural Language Processing (NLP).

## Task Overview

The project was structured around a dataset comprising 50,000 movie reviews sourced from IMDb, each annotated with a sentiment label indicating a positive or negative review. The primary challenge was to design a machine learning model leveraging logistic regression to predict these sentiment labels based on the textual content of the reviews.

### Provided Resources

- **Dataset**: The foundational element of the project was the IMDb movie review dataset, which included a balanced distribution of 50,000 sentiment-tagged reviews.
- **Initial Code Base**: The project commenced with a Python script named `dense_linear_classifier.py`, responsible for several preliminary tasks: loading the dataset, partitioning it into training, validation, and testing subsets, and importing pre-trained word embeddings. These embeddings, crucial for the task, were sourced from the `spacy-2.3.5` package's `en_core_web_md` model and had been pre-filtered to include only those words found within the reviews of the dataset.

### Objective and Responsibilities

The crux of the assignment involved the following key responsibilities:
- **Vector Representation**: Develop a method to transform each review into a vector by aggregating the pre-trained word embeddings. This process required tokenizing the review text and then mapping each token to its corresponding embedding.
- **Model Training and Optimization**: Utilize the vector representations of reviews to train a logistic regression model. A significant part of this task was to experiment with different values of the regularization parameter C to optimize model performance.
- **Evaluation**: Assess the efficacy of the model on a held-out test set, ensuring the chosen regularization parameter yields the best balance between underfitting and overfitting, thereby maximizing model accuracy.

The overarching aim was not only to achieve high accuracy in sentiment classification but also to demonstrate a deep understanding of the logistic regression model, the application of word embeddings in text analysis, and the nuances of model tuning and evaluation.

## Methodology

### Task Overview

The assignment involved developing a sentiment analysis model to classify 50,000 IMDb movie reviews. The model utilized logistic regression combined with pre-trained word embeddings from the `en_core_web_md` model of the `spacy` package, filtered to retain only words present in the reviews.



### Data Preprocessing

The preprocessing phase included several key steps:

- **Shuffling**: Randomly shuffled the dataset to prevent any bias.
- **Splitting**: Divided the dataset into training, validation, and testing sets.
- **Tokenization**: Employed `TreebankWordTokenizer` from the `nltk` package for tokenizing the reviews.
- **Vectorization**: Converted each review into a vector by aggregating the embeddings of the tokenized words. Developed a method to vectorize words not found in the pre-trained embeddings by considering the context provided by adjacent words.

### Model Training and Validation

The logistic regression model was trained using the training set, with the regularization parameter C optimized based on performance on the validation set. A recursive search strategy with early stopping was designed to find the optimal C value, ensuring a balance between underfitting and overfitting.

## Results

The optimal C value was determined to be 49.0, which yielded the highest accuracy on the validation set. The final model, trained with this C value on a dataset combining the training and validation sets, achieved an accuracy of 0.8536 on the test set.

## Discussion

### Skills Demonstrated

- **Natural Language Processing**: Proficiency in text processing, including tokenization and handling of unknown words, was demonstrated through the effective use of word embeddings.
- **Machine Learning**: Applied fundamental machine learning techniques, showcasing an understanding of logistic regression, model training, and hyperparameter tuning.
- **Algorithm Development**: The creation of a custom search strategy for the C value optimization illustrates strong algorithmic thinking and problem-solving skills.
- **Code Efficiency**: The project emphasized efficient and optimized code, particularly in data preprocessing and model training processes.

### Learning Outcomes

- **Advanced NLP Techniques**: Gained practical experience in using tokenization and word embeddings for sentiment analysis.
- **Hyperparameter Optimization**: Learned the significance of hyperparameter tuning in improving model performance and developed a unique approach to it.
- **Problem-Solving Abilities**: Enhanced algorithmic thinking through the design of a custom hyperparameter search technique.
- **Practical Machine Learning Implementation**: Acquired hands-on experience in developing, tuning, and evaluating a logistic regression model for a real-world NLP task.

## Conclusion

This project showcased the application of machine learning and NLP techniques in addressing a sentiment classification challenge, demonstrating competencies in data preprocessing, model development, and hyperparameter optimization. The high accuracy achieved on the test set reflects a comprehensive understanding of the methodologies employed and the ability to apply theoretical concepts to solve practical problems.
