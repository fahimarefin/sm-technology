# Semantic Relationship Classification

## Project Overview

This project develops a machine learning model to classify the semantic relationship between pairs of sentences into one of three classes:

*   **Contradiction (0):** The sentences have opposing meanings.
*   **Neutral (1):** The sentences are related but do not imply one another.
*   **Entailment (2):** One sentence logically follows from the other.

This classification is crucial in various natural language processing (NLP) applications, including question answering, text summarization, and information retrieval.

## 1. Dataset

### 1.1. Description

The dataset used for this project is provided by SM Technology in the `train.csv` file.

### 1.2. File: `train.csv`

*   Contains the labeled training data.
    *   `id`: Unique identifier for each sentence pair.
    *   `premise`: The first sentence in the pair.
    *   `hypothesis`: The second sentence in the pair.
    *   `lang_abv`: Language abbreviation.
    *   `language`: Language of the sentences.
    *   `label`: The relationship classification (0, 1, or 2).

## 2. Implementation Details

### 2.1. Data Loading and Preprocessing:

*   Loading the training data from `train.csv` using pandas.
*   Splitting the training data into training and validation sets (80/20 split) using stratified sampling to maintain class proportions.

### 2.2. Detailed Data Preprocessing Steps:

*   **Lowercasing:** Convert all text to lowercase to ensure consistency.
*   **Punctuation Removal:** Remove punctuation marks from the text.
*   **Stop Word Removal:** Eliminate common words (e.g., "the," "a," "is") that do not carry significant meaning. The `nltk.corpus.stopwords` library is used for this purpose.
*   **Tokenization:** Split the text into individual words or tokens. NLTK's `word_tokenize` function can be used.
*   **Lemmatization/Stemming (Optional):** Reduce words to their base or root form (e.g., "running" to "run"). Lemmatization is generally preferred over stemming as it produces valid words. NLTK's `WordNetLemmatizer` can be used for lemmatization.
*   **Special Character Removal:** Remove any non-alphanumeric characters or special symbols that might interfere with the analysis.

### 2.3. Feature Extraction:

*   **TF-IDF Vectorization:** Converts text data into numerical features using Term Frequency-Inverse Document Frequency. This is used with traditional models like Logistic Regression and Random Forest.
*   **Word2Vec Embeddings:** Uses pre-trained Word2Vec models to generate word embeddings for the premise and hypothesis sentences. These embeddings are then combined (concatenated, subtracted, multiplied) to create feature vectors.
*   **Transformer Models (BERT, XLM-RoBERTa):** Employs pre-trained transformer models to generate contextualized word embeddings. The `transformers` library is used for this purpose. Tokenization is performed using the specific tokenizer associated with the chosen transformer model (e.g., `AutoTokenizer.from_pretrained("bert-base-multilingual-cased")` or `XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")`).

## 3. Model Selection and Training

### 3.1. Traditional Models

These models are commonly used for text classification tasks and are trained on TF-IDF features.

*   **Logistic Regression:** A linear model effective for binary classification problems, known for its high training speed. It is often used for sentiment analysis and other binary classification tasks.
*   **Random Forest:** An ensemble learning method combining multiple decision trees to improve prediction accuracy and robustness. It is widely used for both classification and regression tasks.
*   **Decision Tree:** A tree-based model that is simple to interpret but can suffer from overfitting if not regularized. It is often used as a component in ensemble methods like Random Forest.

### 3.2. Transformer-Based Models

These models leverage pre-trained transformer architectures for advanced text classification tasks.

*   **BERT (Bidirectional Encoder Representations from Transformers):** A pre-trained model fine-tuned for sequence classification tasks. It uses the `AutoModelForSequenceClassification` class from the `transformers` library and the `Trainer` API from Hugging Face for efficient training and evaluation.
*   **XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT approach):** A multilingual model trained on a large corpus of text in multiple languages. It is used with precomputed TF-IDF features and Word2Vec embeddings as input, in addition to raw text data.

### 3.3. Advanced Models

#### 3.3.1. AdvancedNLIModel

This is a custom Artificial Neural Network (ANN) architecture designed for Natural Language Inference (NLI) tasks.

##### 3.3.1.1. Architecture Details

*   **Input Branches:**
    *   **TF-IDF Branch:** A sequential neural network processing TF-IDF features, including linear layers, batch normalization, ReLU activation, and dropout for regularization.
    *   **Word2Vec Branch:** Similar to the TF-IDF branch but processes Word2Vec features.
*   **Feature Fusion:** Concatenates the outputs of the TF-IDF and Word2Vec branches.
*   **Multi-Head Attention:** Employs multi-head attention to model interactions between the combined features.
*   **Classifier:** A deeper sequential neural network that outputs predicted class probabilities. It includes linear layers, batch normalization, ReLU activation, and dropout.

##### 3.3.1.2. Model Configuration

*   **Activation Function:** ReLU (Rectified Linear Unit) is used in the hidden layers.
*   **Regularization:** Dropout is applied after each ReLU activation to prevent overfitting. Batch normalization is also used.
*   **Initialization:** The model's weights are initialized using PyTorch's default initialization.
*   **Loss Function:** Cross-Entropy Loss, potentially weighted to handle class imbalance.
*   **Optimizer:** Adam optimizer is used to update the model's weights during training.
*   **Learning Rate Scheduler:** Not explicitly used but could be added for improved training.

#### 3.3.2. Fine-tuning Google MT5 with LSTM/GRU for Natural Language Inference (NLI)

This project implements and fine-tunes a model that combines the Google MT5 transformer with LSTM or GRU layers for Natural Language Inference (NLI) tasks. It leverages the pre-trained MT5 model for contextual understanding and recurrent layers for sequence-based learning.

##### 3.3.2.1. Model Implementation Details

*   **Model:** `GoogleTransformerLSTMNLI`, a custom architecture combining Google MT5 with LSTM or GRU layers.

    *   **Transformer Layer:** The architecture utilizes the `google/mt5-base` model (or another specified MT5 variant) to generate contextualized embeddings of the input text.
        *   The MT5 model processes the input sequence and outputs the last hidden state. For MT5, only the encoder part is used. The hidden size is dynamically determined from the model config (either `hidden_size` or `d_model` attribute). If neither attribute is available, a default value of 768 is used.
    *   **Recurrent Layer (LSTM or GRU):** An LSTM or GRU layer processes the transformer's output for sequence modeling.
        *   The recurrent layer receives the sequence output from the transformer. The type of recurrent layer (LSTM or GRU) can be configured using the `recurrent_type` parameter.
        *   If `bidirectional` is set to `True`, the recurrent layer is bidirectional, processing the sequence in both forward and backward directions, effectively doubling the number of output features.
        *   The `lstm_units` parameter controls the number of hidden units in the recurrent layer (default: 256).
    *   **Pooling Layer:**
        *   The output of the recurrent layer is pooled using global average pooling (`torch.mean(lstm_output, dim=1)`).
    *   **Classification Layers:** The pooled output is passed through a series of dense layers for classification.
        *   Two dense layers with 512 and 256 units, respectively, are used. Each dense layer is followed by a GELU activation function, layer normalization, and dropout (dropout rate = 0.3). The layers are:
            *   `nn.Linear(lstm_units * 2 if bidirectional else lstm_units, 512)`
            *   `nn.LayerNorm(512)`
            *   `nn.GELU()`
            *   `nn.Dropout(dropout_rate)`
            *   `nn.Linear(512, 256)`
            *   `nn.LayerNorm(256)`
            *   `nn.GELU()`
            *   `nn.Dropout(dropout_rate)`
        *   A final linear layer maps the features to the number of classes (default: 3)
            *   `nn.Linear(256, num_classes)`

##### 3.3.2.2. Model Configuration Details

*   **Activation Function:** GELU (Gaussian Error Linear Unit) is used as the activation function in the dense layers (`F.gelu()`).
*   **Regularization:** Dropout (with a rate of 0.3) is applied after each dense layer to prevent overfitting (`nn.Dropout(dropout_rate)`). Layer normalization is used to stabilize training and improve generalization (`nn.LayerNorm()`).
*   **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
*   **Optimizer:** AdamW optimizer is used to update the model's weights during training (`AdamW`).
*   **Learning Rate Scheduler:** A custom `WarmupLinearSchedule` is implemented to warm up the learning rate linearly during the initial steps and then decay it linearly.
*   **Data Augmentation:** A data augmentation technique (`augment_data`) is applied to the training data to improve model robustness. It performs simple word swaps in the premise and hypothesis.

## 4. Model Evaluation

### 4.1. Evaluation Metrics

*   **Validation Set Evaluation:** The trained model is evaluated on the validation set.
*   **Performance Metrics:** Accuracy, precision, recall, and F1-score are calculated.
*   **Detailed Analysis Tools:** A classification report and confusion matrix are generated for detailed analysis.
*   **ROC AUC (Receiver Operating Characteristic Area Under the Curve):** Used for evaluating the performance of multi-class classification models. Requires binarizing the labels.

## 5. Results

### 5.1. Model Evaluation Results

The model's performance is evaluated based on the validation set. The following metrics are used:

*   **Accuracy:** Overall correctness of the classification.
*   **Precision:** Ability of the model to correctly identify positive instances.
*   **Recall:** Ability of the model to correctly identify positive instances.
*   **F1-score:** Harmonic mean of precision and recall.
*   **Confusion Matrix:** Visualization of the model's performance, showing true positives, true negatives, false positives, and false negatives.
*   **Classification Report:** Detailed performance for each class.
*   **ROC AUC:** Receiver Operating Characteristic Area Under the Curve.

### 5.2. Performance Table

Fill the table below with the correct values after running all the notebooks

| Model                                  | Accuracy | Precision | Recall | F1-Score |
| -------------------------------------- | -------- | --------- | ------ | -------- |
| Google MT5 with LSTM/GRU               | 0.71   | 0.711   | 0.71| 0.71   |
| XLM-RoBERTa                            | 0.71     | 0.71      | 0.71   | 0.71     |
| BERT                                   | 0.67     | 0.67      | 0.67   | 0.67     |
| Custom Artificial Neural Network (ANN) | 0.38   | 0.39      | 0.39   | 0.38     |
| XGBoost                                | 0.38     | 0.38      | 0.38   | 0.37     |
| Random Forest                            | 0.37     | 0.39      | 0.37   | 0.35     |
| Decision Tree                            | 0.36     | 0.37      | 0.36   | 0.36     |


## 6. Usage

### 6.1. Key Libraries Used:

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical computations.
*   `matplotlib`: For data visualization.
*   `seaborn`: For data visualization.
*   `scikit-learn`: For machine learning algorithms (Logistic Regression, Random Forest, Decision Tree), model evaluation, and feature extraction (CountVectorizer, TfidfVectorizer).
*   `nltk`: For natural language processing tasks (stopwords, tokenization, lemmatization).
*   `transformers`: For using pre-trained models like BERT and XLM-RoBERTa.
*   `torch`: For using transformer models.
*   `datasets`: For efficiently handling datasets with transformer models.
*   `huggingface_hub`: For logging into the Hugging Face Hub to save and load models.
*   `tqdm`: For progress bars.

### 6.2. Steps to Run the Code

*I have trained the model's on hugging face cause hugging face provides free gpu to train the llm models.*

*   Install the required packages using `pip install -U pandas numpy torch transformers datasets scikit-learn matplotlib seaborn tqdm huggingface_hub nltk`
*   installed this packages using pip

### 6.3. Data Acquisition:

*   **This step has been completed on Kaggle.** The `train.csv` dataset and the preprocessed data (`multilingual_processed_train_text.csv`, `multilingual_processed_val_text.csv`, `multilingual_train_tfidf_features.npy`, `multilingual_val_tfidf_features.npy`, `multilingual_train_premise_w2v.npy`, `multilingual_train_hypothesis_w2v.npy`, `multilingual_val_premise_w2v.npy`, `multilingual_val_hypothesis_w2v.npy`) are already available in my Kaggle environment.

### 6.4. Code Execution:

*   To ensure the code runs correctly, install the necessary packages. You can do this in a Jupyter Notebook cell by running:

    ```
    !pip install numpy>=1.20.0
    !pip install pandas>=1.3.0
    !pip install torch>=1.9.0
    !pip install scikit-learn>=0.24.0
    !pip install matplotlib>=3.4.0
    !pip install seaborn>=0.11.0
    !pip install tqdm>=4.62.0
    !pip install transformers>=4.11.0
    !pip install datasets>=1.11.0
    !pip install huggingface-hub>=0.0.19
    !pip install nltk>=3.6.0
    !pip install logging>=0.5.1.2
    !pip install datetime
    ```

    Alternatively, you can create a virtual environment (e.g., in VS Code) and install the dependencies using a `requirements.txt` file.  (Note: The exclamation point `!` is only needed when running pip commands directly within a Jupyter Notebook cell.)


**`sm-technology-data-preprocessing-decission-tree-randomforest-XGBoost.ipynb`:** This notebook contains the implementation using traditional machine learning models (Logistic Regression, Random Forest, Decision Tree) with TF-IDF features.
*   **`sm_technology_bert_transformer.ipynb`:** This notebook implements the BERT transformer model. Key steps include:
    *   Loading the pre-trained BERT model and tokenizer.
    *   Preparing the data as Hugging Face Datasets.
    *   Defining training arguments (learning rate, batch size, epochs).
    *   Training the model using the Trainer API.
    *   Evaluating the model and generating metrics (accuracy, classification report, confusion matrix, ROC AUC).
    *   Logging into the Hugging Face Hub and saving the model
    *   and testing model classification result.
    *   The model is trained for 8 epochs.

*   **`sm-tec-xml-roberta-transformer-model.ipynb`:** This notebook implements the XLM-RoBERTa transformer model. Key steps include:
    *   Loading preprocessed data (TF-IDF features, Word2Vec embeddings, processed text).
    *   Loading the XLM-RoBERTa tokenizer and model.
    *   Preparing the data as Hugging Face Datasets.
    *   Defining training arguments (learning rate, batch size, epochs).
    *   Training the model using the Trainer API.
    *   Evaluating the model and generating metrics (accuracy, classification report, confusion matrix, ROC AUC).
    *   Logging into the Hugging Face Hub and saving the model
    *   and testing model classification result.
*   **`sm-technology-data-preprocessing-decission-tree-randomforest-XGBoost.ipynb`:** This notebook contains the implementation using traditional machine learning models (Logistic Regression, Random Forest, Decision Tree) with TF-IDF features.
*   **`sm_technology_bert_transformer.ipynb`:** This notebook implements the BERT transformer model. Key steps include:
    *   Loading the pre-trained BERT model and tokenizer.
    *   Preparing the data as Hugging Face Datasets.
    *   Defining training arguments (learning rate, batch size, epochs).
    *   Training the model using the Trainer API.
    *   Evaluating the model and generating metrics (accuracy, classification report, confusion matrix, ROC AUC).
    *   Logging into the Hugging Face Hub and saving the model
    *   and testing model classification result.
    *   The model is trained for 8 epochs.
*   **`sm-tec-xml-roberta-transformer-model.ipynb`:** This notebook implements the XLM-RoBERTa transformer model. Key steps include:
    *   Loading preprocessed data (TF-IDF features, Word2Vec embeddings, processed text).
    *   Loading the XLM-RoBERTa tokenizer and model.
    *   Preparing the data as Hugging Face Datasets.
    *   Defining training arguments (learning rate, batch size, epochs).
    *   Training the model using the Trainer API.
    *   Evaluating the model and generating metrics (accuracy, classification report, confusion matrix, ROC AUC).
    *   Logging into the Hugging Face Hub and saving the model
    *   and testing model classification result.

*   **`Sm_tec_Custom_Ann.ipynb`:** This notebook implements a custom Artificial Neural Network (ANN) model (`AdvancedNLIModel`) for semantic relationship classification. Key steps include:
    *   Loading preprocessed TF-IDF and Word2Vec features.
    *   Implementing the `AdvancedNLIModel` architecture with TF-IDF and Word2Vec branches, feature fusion, multi-head attention, and a deep classifier network.
    *   Training the custom ANN model.
    *   Evaluating the model and generating metrics (accuracy, precision, recall, F1-score, classification report, confusion matrix, ROC AUC).
    *   Utilizing stratified oversampling to address class imbalance during training.

*   **`sm-tec-lstm-gru-googlemt5.ipynb`:** This notebook focuses on fine-tuning a Google MT5 transformer model combined with LSTM or GRU layers for Natural Language Inference (NLI). Key steps include:
    *   Defining a custom `GoogleTransformerLSTMNLI` model that integrates the `google/mt5-base` transformer with recurrent layers (LSTM or GRU).
    *   Loading preprocessed data (TF-IDF features, Word2Vec embeddings, processed text).
    *   Implementing data augmentation techniques (e.g., word swaps).
    *   Defining and utilizing a `WarmupLinearSchedule` for learning rate scheduling.
    *   Training and evaluating the `GoogleTransformerLSTMNLI` model.
    *   Logging into the Hugging Face Hub and potentially saving the fine-tuned model.


*   Ensure that the file paths in the notebooks (e.g., for loading the dataset) are correct.
*   Execute the cells sequentially to perform data loading, preprocessing, model training, and evaluation.

## 7. Additional Notes

### 7.1. Additional Observations and Notes

*   The project explores different modeling approaches, from traditional machine learning to state-of-the-art transformer models.
*   Hyperparameter tuning can potentially improve model performance. Experiment with different learning rates, batch sizes, and epochs.
*   Consider experimenting with different text preprocessing techniques to optimize feature extraction.
*   The dataset contains sentences in multiple languages. Language-specific preprocessing or multilingual models may improve performance.
*   The notebooks are configured to run on Kaggle with a GPU. Adjust the environment settings if running locally.
*   The XLM-RoBERTa notebook includes code for logging into the Hugging Face Hub and saving the trained model. This allows for easy sharing and deployment of the model.
*   The Word2Vec embeddings were precomputed. Consider exploring different Word2Vec training parameters or using other pre-trained word embeddings.
*   The `Sm_tec_Custom_Ann.ipynb` notebook implements a custom ANN model that combines TF-IDF and Word2Vec features with multi-head attention for improved semantic relationship classification. It addresses class imbalance using stratified oversampling.
*   The `sm-tec-lstm-gru-googlemt5.ipynb` notebook explores fine-tuning a `google/mt5-base` transformer model with LSTM/GRU layers for NLI, including custom data augmentation and a `WarmupLinearSchedule` learning rate scheduler. This demonstrates how to integrate recurrent layers with transformers for sequence modeling.
*   All the models are trained on Kaggle notebooks.
