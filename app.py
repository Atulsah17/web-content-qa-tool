from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('punkt', quiet=True)

app = Flask(__name__)
content_store = {}

def scrape_url(url):
    """Scrape text from a URL using HTTP GET request."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3']))
        return text.strip() or "No readable content found."
    except requests.RequestException as e:
        app.logger.error(f"Failed to scrape {url}: {e}")
        return None

def process_input(input_text, is_url=True):
    """Ingest URLs or raw text into content store."""
    content_store.clear()
    if is_url:
        urls = [url.strip() for url in input_text.splitlines() if url.strip()]
        if not urls:
            return "No valid URLs provided."
        for i, url in enumerate(urls):
            text = scrape_url(url)
            if text:
                content_store[f"doc_{i}"] = text
        if not content_store:
            return "No content could be ingested from the provided URLs."
    else:
        if input_text.strip():
            content_store["doc_0"] = input_text.strip()
        else:
            return "No text provided."
    return None

def answer_question(question):
    """Generate a relevant answer from ingested content."""
    if not content_store:
        return "No content ingested yet. Please submit URLs or text first."
    full_text = ' '.join(content_store.values())
    sentences = nltk.sent_tokenize(full_text)
    if not sentences:
        return "No content could be processed into sentences."
    
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    
    if similarities[best_match_idx] < 0.2:
        return "No sufficiently relevant answer found in the ingested content."
    
    best_sentence = sentences[best_match_idx].strip()
    return (best_sentence[:500].rsplit(' ', 1)[0] + "...") if len(best_sentence) > 500 else best_sentence + "."

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle main page rendering and form submissions."""
    message = answer = ""
    if request.method == 'POST':
        if 'urls_or_text' in request.form:
            input_text = request.form['urls_or_text']
            input_type = request.form.get('input_type', 'urls')
            error = process_input(input_text, is_url=(input_type == 'urls'))
            message = error or f"Ingested {len(content_store)} document(s) successfully."
        elif 'question' in request.form:
            question = request.form['question'].strip()
            answer = answer_question(question) if question else "Please enter a question."
    return render_template('index.html', message=message, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)