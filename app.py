from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF, used for PDF processing
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)

# Load your NER model using the Transformers pipeline for easier handling
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = AutoModelForTokenClassification.from_pretrained('ner_model')
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_text_file(file_path):
    """Extracts text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def highlight_entities(text, results):
    """Highlights named entities in the text."""
    highlighted_text = ""
    last_idx = 0
    for result in results:
        start = result['start']
        end = result['end']
        label = result['entity']
        score = result['score']
        word = result['word']
        tooltip = f"{label} (Score: {score:.2f})"

        # Handle subwords
        if word.startswith("##"):
            word = word[2:]

        highlighted_text += text[last_idx:start]  # Text before the entity
        highlighted_text += f'<span class="entity {label}" data-tooltip="{tooltip}">{word}</span>'
        last_idx = end
    highlighted_text += text[last_idx:]  # Text after the last entity
    return highlighted_text

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the uploading and processing of a PDF or text file."""
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Check the file format
    if file.filename.endswith('.pdf'):
        temp_path = "temp.pdf"
        file.save(temp_path)
        text = extract_text_from_pdf(temp_path)
    elif file.filename.endswith('.txt'):
        temp_path = "temp.txt"
        file.save(temp_path)
        text = extract_text_from_text_file(temp_path)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Process the extracted text
    results = ner_pipeline(text)

    # Format results with highlighted text
    highlighted_text = highlight_entities(text, results)

    # Prepare entity data for the table
    entity_data = [{'word': res['word'], 'entity': res['entity'], 'score': float(res['score'])} for res in results]

    return jsonify({'highlighted_text': highlighted_text, 'entities': entity_data})


if __name__ == '__main__':
    app.run(debug=True)
