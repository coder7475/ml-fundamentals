import re
import string
from bnlp import BasicTokenizer, BengaliStopWords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict

# Initialize Bangla tokenizer and stopwords
bangla_tokenizer = BasicTokenizer()
stopwords = BengaliStopWords().stopwords  # Predefined Bangla stopwords

# Bangla negation words (extend as needed)
negation_words = {'না', 'নয়', 'নাহ', 'নাই'}

def clean_text(text: str) -> str:
    """
    Clean Bangla text by removing URLs, numbers, special characters, and English words.
    Retain Bangla punctuation for sentiment analysis.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers
    text = re.sub(r'[০-৯0-9]', '', text)
    # Remove special characters except Bangla punctuation (।, ?, !)
    text = re.sub(r'[^\u0980-\u09FF।?! ]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower()

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize Bangla text into words using bnlp_toolkit.
    """
    return bangla_tokenizer.tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove Bangla stopwords from tokenized text.
    """
    return [token for token in tokens if token not in stopwords]

def handle_negations(tokens: List[str]) -> List[str]:
    """
    Mark negated words in Bangla text by appending '_NEG' to words following negation.
    """
    processed_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i + 1 < len(tokens):
            processed_tokens.append(tokens[i])  # Keep negation word
            processed_tokens.append(tokens[i + 1] + '_NEG')  # Mark next word
            i += 2
        else:
            processed_tokens.append(tokens[i])
            i += 1
    return processed_tokens

def normalize_text(tokens: List[str]) -> List[str]:
    """
    Normalize Bangla tokens (basic normalization, as lemmatization is limited).
    Replace common variations (e.g., dialectal forms). Extend as needed.
    """
    # Example normalization dictionary (extend for your use case)
    normalization_dict = {
        'খুবই': 'খুব',
        'ভালোবাসি': 'ভালবাসা'
    }
    return [normalization_dict.get(token, token) for token in tokens]

def get_tfidf_features(texts: List[str]) -> np.ndarray:
    """
    Convert cleaned Bangla texts to TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer

def get_bert_embeddings(texts: List[str], model_name: str = 'sagorsarker/bangla-bert-base') -> np.ndarray:
    """
    Get BERT embeddings for Bangla text using a pre-trained Bangla BERT model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

def encode_labels(labels: List[str], label_map: Dict[str, int] = None) -> List[int]:
    """
    Encode sentiment labels to numerical values.
    """
    if label_map is None:
        label_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    return [label_map[label] for label in labels]

def preprocess_bangla_text(texts: List[str], use_bert: bool = False) -> Dict:
    """
    Full pre-processing pipeline for Bangla sentiment analysis.
    Args:
        texts: List of raw Bangla texts.
        use_bert: If True, use Bangla BERT embeddings; else, use TF-IDF.
    Returns:
        Dictionary with processed texts or features and tokens.
    """
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize_text(text) for text in cleaned_texts]
    normalized_texts = [normalize_text(tokens) for tokens in tokenized_texts]
    negated_texts = [handle_negations(tokens) for tokens in normalized_texts]
    filtered_texts = [remove_stopwords(tokens) for tokens in negated_texts]
    
    # Join tokens back for feature extraction
    processed_texts = [' '.join(tokens) for tokens in filtered_texts]
    
    if use_bert:
        features = get_bert_embeddings(processed_texts)
    else:
        features, vectorizer = get_tfidf_features(processed_texts)
    
    return {
        'processed_texts': processed_texts,
        'features': features,
        'tokens': filtered_texts
    }

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "এই মুভিটি খুব ভালো ছিল।",
        "আমি এটা একদম পছন্দ করিনি।",
        "সেবাটি সাধারণ ছিল, না ভালো না খারাপ।"
    ]
    sample_labels = ['positive', 'negative', 'neutral']
    
    # Pre-process texts
    result = preprocess_bangla_text(sample_texts, use_bert=False)
    
    # Encode labels
    encoded_labels = encode_labels(sample_labels)
    
    print("Processed Texts:", result['processed_texts'])
    print("Features Shape:", result['features'].shape)
    print("Encoded Labels:", encoded_labels)