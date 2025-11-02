from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time
import sys

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
PORT = 5001 # Run on a different port than the main Go app
# ---

app = Flask(__name__)

# Load the model ONCE at startup
print(f"Loading sentence transformer model '{MODEL_NAME}' for NLP service...")
start_time = time.time()
try:
    embedding_model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load model. ---")
    print(f"Ensure 'sentence-transformers' and 'torch' are installed: pip install sentence-transformers torch")
    print(f"Error: {e}")
    embedding_model = None
    sys.exit(1) # Exit if model fails to load

@app.route('/encode', methods=['POST'])
def encode_query():
    """
    Receives a JSON body like {"query": "..."} and returns
    the embedding vector as {"vector": [...]}.
    """
    if embedding_model is None:
        return jsonify({"error": "NLP model is not loaded."}), 500
        
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No 'query' field provided."}), 400
        
    try:
        # Encode the query
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        # Convert numpy array to a simple list for JSON
        vector_list = query_embedding[0].tolist() 
        
        return jsonify({"vector": vector_list})
        
    except Exception as e:
        print(f"--- ERROR during query encoding: {e} ---")
        return jsonify({"error": "Failed to encode query."}), 500

if __name__ == '__main__':
    print(f"--- Starting Python NLP Microservice on http://127.0.0.1:{PORT} ---")
    # Bind to 0.0.0.0 to be accessible from other services (like Go)
    app.run(host='0.0.0.0', port=PORT)