import os
import math # Added for ceiling function in scaling
from dotenv import load_dotenv
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer 


# --- Page Configuration (THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="AI Image-Text Relevance Scorer")

# Load environment variables
load_dotenv()

# --- Configuration & Model Loading ---
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe') # Uses a specific name TESSERACT_PATH
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# Configure Tesseract path
try:
    # Check if the path exists before setting
    if os.path.exists(TESSERACT_PATH):
         pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
         st.warning(f"Tesseract path specified ({TESSERACT_PATH}) but not found. OCR might fail.")
except Exception as e:
    st.warning(f"Could not configure Tesseract path: {e}")

# Initialize models using Streamlit's caching
@st.cache_resource
def load_models():
    st.write() 
    try:
        # BLIP for image captioning
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Similarity model (Sentence Transformer)
        sim_model = SentenceTransformer('all-MiniLM-L6-v2')

        return {
            'blip_processor': processor,
            'blip_model': blip_model,
            'sim_model': sim_model
        }
    except Exception as e:
        st.error(f"Fatal Error: Could not load AI models. {e}")
        return None # Returns None or raise error if models are critical


# --- Helper Functions ---

def scrape_google_images(query, num_images=100):
    """Scrapes Google Images using API credentials from .env"""
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        st.error("Google API credentials not configured. Please set GOOGLE_API_KEY and SEARCH_ENGINE_ID in .env file")
        return []

    base_url = "https://www.googleapis.com/customsearch/v1"
    results = []
    num_to_fetch = min(num_images, 100) # Google API limit is 100 results total (10 per page)

    try:
        # API allows max 10 results per call, max 100 total. Paginate accordingly.
        for start_index in range(1, num_to_fetch + 1, 10):
            num_in_call = min(10, num_to_fetch - start_index + 1) # Adjust num for last page
            if num_in_call <= 0: break

            params = {
                'q': query,
                'key': GOOGLE_API_KEY,
                'cx': SEARCH_ENGINE_ID,
                'searchType': 'image',
                'num': num_in_call,
                'start': start_index
            }
            response = requests.get(base_url, params=params, timeout=15) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses
            data = response.json()

            if 'items' not in data:
                # No more items found
                break

            for item in data['items']:
                 # Store relevant info, ensuring keys exist
                results.append({
                    'url': item.get('link'),
                    'google_desc': (item.get('title', '') + " " + item.get('snippet', '')).strip(), # Use specific key name
                    'context_link': item.get('image', {}).get('contextLink'), # Use specific key name
                    'original_rank': len(results) + 1
                })

                # Stops if we reached the desired number
                if len(results) >= num_to_fetch:
                    break
            if len(results) >= num_to_fetch:
                 break # Exit outer loop too

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching images from Google API: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during image scraping: {str(e)}")

    return results # Return collected results


def download_image(url):
    """Download and return PIL Image"""
    if not url: return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        # Check content type
        content_type = response.headers.get('content-type')
        if not content_type or not content_type.startswith('image/'):
            print(f"Skipping non-image URL: {url} (Content-Type: {content_type})")
            return None
        img = Image.open(BytesIO(response.content))
        # Ensure image has 3 channels (RGB) for compatibility with some models/OpenCV
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except requests.exceptions.Timeout:
        st.warning(f"Timeout downloading {url}")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to download {url}: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"Failed to process image from {url}: {str(e)}")
        return None


def generate_blip_caption(image, models):
    """Generate image caption using BLIP"""
    try:
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = models['blip_processor'](image, return_tensors="pt")
        # Adjust generation parameters if needed (e.g., max_length)
        outputs = models['blip_model'].generate(**inputs, max_length=75)
        caption = models['blip_processor'].decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        st.warning(f"BLIP caption generation failed: {str(e)}")
        return "Caption generation failed."


def extract_text_with_tesseract(image):
    """Extract text using Tesseract OCR"""
    try:
        # Convert PIL Image to OpenCV format (NumPy array)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Optional: Add preprocessing steps here if needed (e.g., grayscale, thresholding)
        # img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(img_cv) # Pass OpenCV image to Tesseract
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract not found or not configured correctly. OCR is disabled.")
        return "Tesseract not found."
    except Exception as e:
        st.warning(f"OCR failed: {str(e)}")
        return ""


def calculate_cosine_similarity(text1: str, text2: str, sim_model) -> float:
    """Calculates cosine similarity between two texts using Sentence Transformer."""
    if not text1 or not text2 or not sim_model:
        return 0.0
    try:
        # Encode texts into embeddings
        embeddings = sim_model.encode([text1, text2], convert_to_tensor=False) # Get numpy arrays
        # Calculate cosine similarity (ensure embeddings are 1D arrays)
        if embeddings[0].ndim == 1 and embeddings[1].ndim == 1:
            cosine_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        else: # Handle potential issues with embedding dimensions if model changes
            cosine_sim = 0.0
            print(f"Warning: Unexpected embedding dimensions. Shape 1: {embeddings[0].shape}, Shape 2: {embeddings[1].shape}")
        # Clip similarity score between 0 and 1 (sometimes float precision can go slightly outside)
        return max(0.0, min(1.0, float(cosine_sim)))
    except Exception as e:
        st.warning(f"Error calculating cosine similarity: {e}")
        return 0.0

def scale_relevance_score(similarity: float, min_score: int = 1, max_score: int = 10) -> int:
    """Scales a similarity score (0-1) to a relevance score (e.g., 1-10)."""
    if similarity <= 0:
        return min_score
    # Use ceiling to give a slight boost, ensuring 1.0 similarity reaches max_score
    scaled = min_score + (max_score - min_score) * similarity
    return max(min_score, min(max_score, math.ceil(scaled)))

# --- Main Application Logic ---
def main():
    st.title("AI Image-Text Relevance Scorer")
    st.markdown("""
    Enter a text query. This tool fetches images from Google, analyzes them using AI
    (image captioning and text extraction), calculates relevance based on semantic similarity,
    and displays the results.
    """)

    # Load models (will be cached after first run)
    models = load_models()
    if models is None:
        st.error("Failed to load AI models. Application cannot continue.")
        st.stop() # Stop execution if models are essential

    # --- User Input Area ---
    query = st.text_input("Enter your search query:", key="query_input")

    # Use a slider instead of number input
    num_images_to_process = st.slider(
        "Select number of images to analyze:",
        min_value=5,   # Minimum number of images
        max_value=100, # Maximum number of images (Google API limit)
        value=20,      # Default value when the app loads
        step=5,        # Increment step for the slider
        key="num_images_slider" # Unique key for the widget
    )
    # Display the selected value next to the slider (optional, but helpful)
    st.caption(f"Will fetch and analyze the top {num_images_to_process} images.")

    search_button = st.button("🔍 Search and Analyze Images", key="search_button")

    # Initialize session state for results
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []

    # --- Processing Logic ---
    if search_button and query:
        # Check API keys again before proceeding
        if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            st.error("Missing Google API credentials. Please add GOOGLE_API_KEY and SEARCH_ENGINE_ID to your .env file.")
            st.code("""Example .env content:
GOOGLE_API_KEY=AIzaSy...
SEARCH_ENGINE_ID=abc123...
# Optional: TESSERACT_PATH=/path/to/tesseract.exe""")
        else:
            st.session_state.processed_results = [] # Clear previous results
            with st.spinner(f"Searching for '{query}' and analyzing top {num_images_to_process} images..."):
                # 1. Scrape image URLs and metadata
                # Pass the value from the slider to the scraping function
                image_data = scrape_google_images(query, num_images_to_process)

                if not image_data:
                    st.warning("No images found via Google Search for this query, or an API error occurred.")
                else:
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    # Use the actual number of images returned by the API for progress calculation
                    total_images_to_process_api = len(image_data)

                    for i, img_info in enumerate(image_data):
                        current_progress = (i + 1) / total_images_to_process_api
                        status_text.text(f"Processing image {i+1}/{total_images_to_process_api} (Rank: {img_info['original_rank']})...")
                        progress_bar.progress(current_progress)

                        # 2. Download Image
                        img_pil = download_image(img_info['url'])
                        if img_pil is None:
                            print(f"Skipping image {i+1} (Rank {img_info['original_rank']}) due to download/processing failure.")
                            continue # Skip to next image if download fails

                        try:
                            # 3. Analyze Image: Captioning & OCR
                            blip_caption = generate_blip_caption(img_pil, models)
                            ocr_text = extract_text_with_tesseract(img_pil)

                            # 4. Combine Text Representations
                            combined_text = f"Source Description: {img_info['google_desc']}. Image Content: {blip_caption}. Text in Image: {ocr_text}".strip()

                            # 5. Calculate Relevance
                            raw_similarity_score = calculate_cosine_similarity(query, combined_text, models['sim_model'])
                            scaled_relevance_score = scale_relevance_score(raw_similarity_score)

                            # Store all relevant data
                            results_list.append({
                                'pil_image': img_pil,
                                'image_url': img_info['url'],
                                'original_rank': img_info['original_rank'],
                                'google_desc': img_info['google_desc'],
                                'blip_caption': blip_caption,
                                'ocr_text': ocr_text,
                                'combined_text': combined_text,
                                'similarity_score': raw_similarity_score,
                                'relevance_score': scaled_relevance_score,
                                'context_link': img_info['context_link']
                            })
                            print(f"  Processed Rank {img_info['original_rank']}: Sim={raw_similarity_score:.3f}, Rel={scaled_relevance_score}")
                        
                        except Exception as e:
                            st.warning(f"Error processing image at URL {img_info['url']} (Rank {img_info['original_rank']}): {str(e)}")
                            continue

                    st.session_state.processed_results = results_list
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Analysis complete! Processed {len(st.session_state.processed_results)} images.")

    # --- Display Results ---
    # (The rest of the display logic remains the same as the previous version)
    if st.session_state.processed_results:
        st.markdown("---")
        st.subheader("Analysis Results")

        # Use tabs for different display modes
        tab1, tab2 = st.tabs(["**Original Order View**","**Relevance Sorted View**"])

        # Data for display (use the list from session state)
        results_to_display = st.session_state.processed_results


        # Original Order Tab
        with tab1:
            st.markdown("Images sorted by their original rank in Google Search results.")
            original_order_results = sorted(results_to_display, key=lambda x: x['original_rank'])

            if not original_order_results:
                st.info("No results to display.")
            else:
                num_columns_orig = 4
                cols_orig = st.columns(num_columns_orig)
                for idx, res in enumerate(original_order_results):
                    col = cols_orig[idx % num_columns_orig]
                    with col:
                        img_caption = f"Orig Rank: {res['original_rank']} | Relevance: {res['relevance_score']}/10"
                        st.image(res['pil_image'], caption=img_caption, use_column_width=True)
                        with st.expander("Show Details"):
                            st.markdown(f"**Original Google Rank:** {res['original_rank']}")
                            st.markdown(f"**Relevance Score (1-10):** {res['relevance_score']}")
                            st.markdown(f"**Similarity Score (0-1):** {res['similarity_score']:.4f}")
                            st.markdown(f"**Generated Caption (BLIP):** {res['blip_caption']}")
                            st.markdown(f"**Extracted Text (OCR):** {res['ocr_text'] if res['ocr_text'] else 'None detected'}")
                            st.markdown(f"**Google Description:** {res['google_desc']}")
                            if res['context_link']:
                                st.markdown(f"**[Source Page Link]({res['context_link']})**", unsafe_allow_html=True)
                            else:
                                st.markdown("**Source Page Link:** Not available")

        # Relevance Sorted Tab
        with tab2:
            st.markdown("Images sorted by calculated relevance score (highest first).")
            sorted_results = sorted(results_to_display, key=lambda x: x['similarity_score'], reverse=True)

            if not sorted_results:
                st.info("No results to display.")
            else:
                num_columns_sorted = 4 # Adjust number of columns
                cols_sorted = st.columns(num_columns_sorted)
                for idx, res in enumerate(sorted_results):
                    col = cols_sorted[idx % num_columns_sorted]
                    with col:
                        img_caption = f"Relevance: {res['relevance_score']}/10 | Orig Rank: #{res['original_rank']}"
                        st.image(res['pil_image'], caption=img_caption, use_column_width=True)
                        with st.expander("Show Details"):
                            st.markdown(f"**Relevance Rank:** {idx + 1}")
                            st.markdown(f"**Original Google Rank:** {res['original_rank']}")
                            st.markdown(f"**Relevance Score (1-10):** {res['relevance_score']}")
                            st.markdown(f"**Similarity Score (0-1):** {res['similarity_score']:.4f}")
                            st.markdown(f"**Generated Caption (BLIP):** {res['blip_caption']}")
                            st.markdown(f"**Extracted Text (OCR):** {res['ocr_text'] if res['ocr_text'] else 'None detected'}")
                            st.markdown(f"**Google Description:** {res['google_desc']}")
                            if res['context_link']:
                                st.markdown(f"**[Source Page Link]({res['context_link']})**", unsafe_allow_html=True)
                            else:
                                st.markdown("**Source Page Link:** Not available")

    
    elif search_button and not query:
        st.warning("Please enter a search query.")

    # --- Footer/Info ---
    st.markdown("---")
    st.caption("Uses Google Custom Search, BLIP,Tesseract and Sentence Transformers")


if __name__ == "__main__":
    main()