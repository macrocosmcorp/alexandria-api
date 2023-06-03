import logging
import os

import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from InstructorEmbedding import INSTRUCTOR

# Load .env file. By default, it looks for the .env file in the same directory as the script being run.
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check for necessary environment variables
if not os.getenv('PINECONE_API_KEY'):
    raise ValueError("Environment variable PINECONE_API_KEY is not set")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = INSTRUCTOR('hkunlp/instructor-xl', device=device)

app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search_embeddings():
    logging.info("Received request for search_embeddings")
    data = request.get_json(force=True)

    if 'embedding' not in data:
        return jsonify({'error': 'No embedding provided. Please provide an "embedding" value in the request body.'}), 400

    embedding = data['embedding']
    topK = data.get('topK', 10)

    pinecone_url = 'https://abstracts-97548a4.svc.us-central1-gcp.pinecone.io/query'
    headers = {
        'Api-Key': os.getenv('PINECONE_API_KEY'),
        'accept': 'application/json',
        'content-type': 'application/json',
    }

    body = {
        'vector': embedding,
        'topK': topK,
        'includeValues': True,
        'includeMetadata': True,
    }

    try:
        response = requests.post(pinecone_url, json=body, headers=headers)
        response.raise_for_status()  # Will raise an exception for 4xx and 5xx status codes
    except requests.RequestException as e:
        logging.error(f"Request to Pinecone failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

    response_data = response.json()

    results = [
        {
            'id': match['id'],
            'score': match['score'],
            'metadata': match['metadata'],
        }
        for match in response_data['matches']
    ]

    return jsonify({'results': results})


@app.route('/embed', methods=['POST'])
def embed_text():
    logging.info("Received request for embed_text")
    data = request.get_json(force=True)

    if 'text' not in data:
        return jsonify({'error': 'No text provided. Please provide a "text" value in the request body.'}), 400

    text = data['text']
    embedding = model.encode(text)

    embedding_list = embedding.tolist()

    return jsonify({'embedding': embedding_list})


@app.route('/search_abstracts', methods=['POST'])
def search_text():
    logging.info("Received request for search_text")
    data = request.get_json(force=True)

    if 'text' not in data:
        return jsonify({'error': 'No text provided. Please provide a "text" value in the request body.'}), 400

    text = data['text']
    print("Running encode on text: ", data['text'])
    embedding = model.encode(
        [["Represent the Research Paper query for retrieving related Research Paper abstracts; Input:", text]])
    embedding_list = embedding.tolist()
    print("Done encoding text for text: ", data['text'])

    topK = data.get('topK', 10)

    pinecone_url = 'https://abstracts-97548a4.svc.us-central1-gcp.pinecone.io/query'
    headers = {
        'Api-Key': os.getenv('PINECONE_API_KEY'),
        'accept': 'application/json',
        'content-type': 'application/json',
    }

    body = {
        'vector': embedding_list,
        'topK': topK,
        'includeValues': True,
        'includeMetadata': True,
    }

    try:
        response = requests.post(pinecone_url, json=body, headers=headers)
        response.raise_for_status()  # Will raise an exception for 4xx and 5xx status codes
    except requests.RequestException as e:
        logging.error(f"Request to Pinecone failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

    response_data = response.json()

    results = [
        {
            'id': match['id'],
            'score': match['score'],
            'metadata': match['metadata'],
        }
        for match in response_data['matches']
    ]

    return jsonify({'results': results})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'You tried to reach %s' % path


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
    logging.info(
        f"Server running. Pinecone API key: {os.getenv('PINECONE_API_KEY')}")
