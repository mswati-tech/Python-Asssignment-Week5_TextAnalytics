"""Assuming that all the libraries are downloaded and installed otherwise
adding exception handling code block as the code below
try:
    nltk.download('all')
except Exception as e:
    print(f"[Error] Failed to download NLTK resources: {e}")"""

# Importing all the essential libraries that will be required
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report
import csv
import os
import re
import string
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

"""The key is to clean the datasets to the best so that there is less noise and the models can be implemented effectively.
Therefore, at the beginning defining a method to clean the CSV files by creating an object in the main program.
Using OOP concept."""

class col_cleaner:
  def __init__(self,dataset_col,extra_stopwords=None): # Using dunder method __init__

    """Inititializing with the column that needs to be cleaned in the datasets.
    Initializing other classes from other Python libraries/modules"""

    self.dataset_col = dataset_col

    try: # If the libraries/modules are not installed properly
      self.stopwords = set(stopwords.words('english'))
      if extra_stopwords:
        self.stopwords.update(extra_stopwords)
      self.lemmatizer = WordNetLemmatizer()
      self.stem = PorterStemmer()

    except Exception as e:
      raise RuntimeError(f"[Error] Failed to initialize the Python libraries required to pre-proessing the datasets: {e}")

  def preprocess_text(self,text): # Performing text preprocessing
    try:
      if not text or pd.isnull(text): # Returning an empty dataframe if the passed dataframe is empty
        return ""
        print(df[self.dataset_col].isnull().sum()) # Checking NaN values

      # Converting the text to lowercase
      text = str(text).lower()

      # Performing tokenization
      tokens = word_tokenize(text) #Split

      # Removing unwanted token like 'SDATA'
      tokens = [word for word in tokens if 'sdata' not in word]

      # Removing unwanted token like 'EDATA'
      tokens = [word for word in tokens if 'edata' not in word]

      # Extracting only words and removing punctuations
      clean_tokens = [re.sub(r'[^a-zA-Z]','', word) for word in tokens]
      clean_tokens = [word for word in clean_tokens if word !='']

      # Explicitly removing any expression containing "newlinetoken"
      target_regex = r"newlinetoken"
      target_regex_tokens = [re.sub((rf"\b\w*{target_regex}\w*\b"),'',word) for word in clean_tokens]

      # Removing stopwords
      filtered_tokens = [word for word in target_regex_tokens if word not in self.stopwords]

      # Performing lemmatization
      lemma_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]

      # Performing stemming for words ending with 'ing'
      stem_tokens = [self.stem.stem(word) if word.endswith("ing") else word
                      for word in lemma_tokens] 

      return " ".join(lemma_tokens) #Returning the string on which the ML models will be implemented

    except Exception as e:
      print(f"The text pre-processing could not be performed: {e}")
      return " "

  def clean_csv(self,file_pathname):
    # Reads the file, calls the function to clean the column, and returns the cleaned dataframe
    try:
      df = pd.read_csv(file_pathname)
    
    except FileNotFoundError:
      raise FileNotFoundError(f"[Error] File Not Found: {file_pathname}")
    
    except Exception as e:
      raise RuntimeError(f"[Error] Failed to Read CSV: {file_pathname}")

    if self.dataset_col not in df.columns:
      raise ValueError(f"[Error] Column '{self.dataset_col}' not found in {file_pathname}")

    try:
      df[self.dataset_col] = df[self.dataset_col].apply(self.preprocess_text)
    
    except Exception as e:
      raise RuntimeError(f"[Error] Failed during text cleaning: {e}")

    return df

  def refine_cleaned_csv(self, train_file, valid_file, threshold=0.0005):

    # Refines cleaning by identifying low-importance words from the cleaned train and valid dataset
        
    # Loading train and valid
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)

    X_train = train_df[self.dataset_col].astype(str)
    y_train = train_df["toxicity"]

    X_valid = valid_df[self.dataset_col].astype(str)
    y_valid = valid_df["toxicity"]

    # Vectorizing
    vectorizer = TfidfVectorizer(max_features=20000) #Keep only the most informative features
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    # Training model
    model = LogisticRegression(max_iter=1000) #The number of iterations for best fitting
    model.fit(X_train_vec, y_train)

    # Identifying low-importance words
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    low_importance_mask = (coef > -threshold) & (coef < threshold) # The co-efficient that is between -0.0005 to 0.0005
    low_importance_words = feature_names[low_importance_mask]

    print(f"[INFO] Found {len(low_importance_words)} low-importance words")
    pd.Series(low_importance_words).to_csv("low_importance_words.csv", index=False)

    # Creating new cleaner with extended stopwords
    extra_stopwords = set(low_importance_words)
    refined_cleaner = col_cleaner(dataset_col=self.dataset_col, extra_stopwords=extra_stopwords)

    # Re-cleaning datasets
    for fname in [train_file, valid_file, "test_cleaned.csv"]:
      df_cleaned = refined_cleaner.clean_csv(fname)
      df_cleaned.to_csv(fname.replace(".csv", "_refined.csv"), index=False)
      print(f"[INFO] {fname} refined and saved.")

# Function for Logistic Regression (Discriminative)
def run_logistic_regression(train_file, valid_file, test_file):
  # Loading data
  train_df = pd.read_csv(train_file)
  valid_df = pd.read_csv(valid_file)
  test_df = pd.read_csv(test_file)

  # Vectorizing
  vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
  X_train = vectorizer.fit_transform(train_df["comment"].fillna("")) #fillna("") because it was giving a fatal error whenever it encountered NaN values
  y_train = train_df["toxicity"]

  X_valid = vectorizer.transform(valid_df["comment"].fillna(""))  #fillna("") because it was giving a fatal error whenever it encountered NaN values
  y_valid = valid_df["toxicity"]

  # Training model
  model = LogisticRegression(max_iter=1000, class_weight='balanced')
  """Earlier I had not given class_weight as 'balanced' though I could see that there were many classes with 0 values and fewer classes with 1 values
  So my models were specific to the classes that were non-toxic (0 values) but could not sensitive to the toxic classes (1 values). So to balance the "imbalanced
  dataset" I had to provide weights"""
  model.fit(X_train, y_train)

  # Validating
  y_pred = model.predict(X_valid)
  print("\n[Logistic Regression] Performance on Validation Set:")
  print("Accuracy:", accuracy_score(y_valid, y_pred))
  print("F1-score:", f1_score(y_valid, y_pred))
  print(classification_report(y_valid, y_pred))

  # Predicting on test data
  X_test = vectorizer.transform(test_df["comment"].fillna(""))  #fillna("") because it was giving a fatal error whenever it encountered NaN values
  test_df["out_label_model_Dis"] = model.predict(X_test)

  # Saving updated file
  test_df.to_csv(test_file, index=False)
  print(f"[INFO] Logistic Regression predictions saved to {test_file}")

# Function for Naive Bayes (Generative)
def run_naive_bayes(train_file, valid_file, test_file):
  # Loading data
  train_df = pd.read_csv(train_file)
  valid_df = pd.read_csv(valid_file)
  test_df = pd.read_csv(test_file)

  # Vectorizing
  vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
  X_train = vectorizer.fit_transform(train_df["comment"].fillna(""))  #fillna("") because it was giving a fatal error whenever it encountered NaN values
  y_train = train_df["toxicity"]

  """Earlier I had not assigned weights though I could see that there were many classes with 0 values and fewer classes with 1 values
  So my models were specific to the classes that were non-toxic (0 values) but could not sensitive to the toxic classes (1 values). So to balance the "imbalanced
  dataset" I had to provide weights using Oversampling of the minority class"""

  # Oversampling the minority class
  ros = RandomOverSampler(random_state=42)
  X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

  X_valid = vectorizer.transform(valid_df["comment"].fillna(""))  #fillna("") because it was giving a fatal error whenever it encountered NaN values
  y_valid = valid_df["toxicity"]

  # Training model
  model = MultinomialNB()
  model.fit(X_train_res, y_train_res)

  # Validating
  y_pred = model.predict(X_valid)
  print("\n[Naive Bayes] Performance on Validation Set:")
  print("Accuracy:", accuracy_score(y_valid, y_pred))
  print("F1-score:", f1_score(y_valid, y_pred))
  print(classification_report(y_valid, y_pred))

  # Predicting on test data
  X_test = vectorizer.transform(test_df["comment"].fillna(""))
  test_df["out_label_model_Gen"] = model.predict(X_test)

  # Saving updated file
  test_df.to_csv(test_file, index=False)
  print(f"[INFO] Naive Bayes predictions saved to {test_file}")

# Main program

def main():
  try:

    # Creating cleaner object for col_cleaner class
    cleaner = col_cleaner(dataset_col="comment")

    # Cleaning train.csv
    train_df = cleaner.clean_csv("train.csv")
    train_df.to_csv('train_cleaned.csv', index = False)
    print("[Info] Trained Data cleaned successfully")

    # Cleaning valid.csv
    valid_df = cleaner.clean_csv("valid.csv")
    valid_df.to_csv('valid_cleaned.csv', index = False)
    print("[Info] Valid Data cleaned successfully")

    # Cleaning test.csv
    test_df = cleaner.clean_csv("test.csv")
    test_df.to_csv('test_cleaned.csv', index = False)
    print("[Info] Test Data cleaned successfully")

    """After the files were cleaned, there were words like padding, style, edit, role, etc. that
    do not play a part in deciding toxicity, thererfore it is important to keep the most 
    informative comment words that actually play a part in deciding toxicity"""

    """Refining using Logistic Regression where extract features and decide whether it is toxic or non-toxic
    Threshold experimented between 0 to 0.05"""

    cleaner.refine_cleaned_csv("train_cleaned.csv","valid_cleaned.csv", threshold = 0.0005)

    """After refining the datasets and experimenting with the threshold and max_features, I manually checked the files and 
    I was happy with the cleaning which is when I implemented the NB and Logisitic Regression models"""

    train_path = "train_cleaned_refined.csv"
    valid_path = "valid_cleaned_refined.csv"
    test_path = "test_cleaned_refined.csv"  # will get updated twice

    # Ensuring refined files exist before proceeding
    for f in [train_path, valid_path, test_path]:
      if not os.path.exists(f):
        raise FileNotFoundError(f"[ERROR] Expected file not found: {f}")

    # First ran Naive Bayes so file is created with Generative predictions
    run_naive_bayes(train_path, valid_path, test_path)

    # Then ran Logistic Regression to add Discriminative predictions
    run_logistic_regression(train_path, valid_path, test_path)

    print("[INFO] All steps completed successfully.")

  except Exception as e:
    print(f"[Fatal Error] {e}")

# Running the main() function"""

main()











