# AI Image-Text Relevance Scorer

This Streamlit application retrieves images from Google based on a search query, generates captions using the BLIP image captioning model, extracts visible text using OCR (Tesseract), and evaluates the relevance of each image to the input query using semantic similarity scoring.

## Features

- Google Image Search using Custom Search API
- Image captioning with Salesforce BLIP
- OCR-based text extraction using Tesseract
- Semantic similarity scoring using Sentence Transformers
- Weighted relevance scoring and display
- Interactive Streamlit web interface

## Requirements

- Python 3.8 or higher
- Google API key and Custom Search Engine ID
- Tesseract OCR installed on your system
- Create .env file with all the apis 

## Installation

### 1. Clone the Repository

'''bash
git clone https://github.com/yourusername/AI-Image_Text-Relevance-Scorer.git
cd image-text-relevance-app'''

###  2. Install Dependencies 
pip install -r requirements.txt 

Once setup is complete, launch the streamlit app:
''' bash 
streamlit run app.py
'''

### File Structure
Ai-image-text-relevance-scorer/
├── app.py
├── requirements.txt
└── .env

### Credits 
1. Salesforce BLIP
2. HuggingFace Transformers
3. Tesseract OCR
4. Streamlit 
