# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:54:59 2024

@author: K Saran
"""

C:/Users/K Saran/Downloads/project python/project python.pdf
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Join tokens into a string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Provide the PDF file path here
pdf_path = 'C:/Users/K Saran/Downloads/project python/project python.pdf'

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Preprocess text
cleaned_text = preprocess_text(pdf_text)

# Display preprocessed text
print("Preprocessed Text:")
print(cleaned_text)









import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content.
    """
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def preprocess_text(text):
    """
    Preprocess text by removing punctuation, converting to lowercase, and tokenizing.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Join tokens into a string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Provide the PDF file path here
pdf_path = 'C:/Users/K Saran/Downloads/project python/project python.pdf'

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

if pdf_text:
    # Preprocess text
    cleaned_text = preprocess_text(pdf_text)

    # Find the number of tokens in the preprocessed text
    num_tokens = len(cleaned_text.split())

    # Find the number of characters in the original text extracted from the PDF
    num_characters = len(pdf_text)

    # Display preprocessed text
    print("Preprocessed Text:")
    print(cleaned_text)
    
    # Display the results
    print("Number of tokens in preprocessed text:", num_tokens)
    print("Number of characters in original text:", num_characters)
else:
    print("No text extracted from the PDF. Please check the file path and try again.")