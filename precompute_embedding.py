import csv
import json
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
EVENTS_FILE = 'events.csv'
OUTPUT_FILE = 'event_embeddings.json'
# ---

def precompute():
    """
    Loads all event labels from events.csv, computes their embeddings
    using SentenceTransformer, and saves them to a JSON file.
    """
    print(f"Loading model '{MODEL_NAME}'... (This may take a moment)")
    start_time = time.time()
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load model. ---")
        print(f"Ensure 'sentence-transformers' and 'torch' are installed: pip install sentence-transformers torch")
        print(f"Error: {e}")
        return

    labels_to_encode = []
    event_id_map = [] # To store the ID corresponding to each label

    print(f"Reading events from {EVENTS_FILE}...")
    try:
        with open(EVENTS_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if 'Event_ID' not in reader.fieldnames or 'Event_Label' not in reader.fieldnames:
                raise KeyError("events.csv must have 'Event_ID' and 'Event_Label' columns.")
            
            for row in reader:
                event_id = row.get('Event_ID')
                label = row.get('Event_Label')
                if event_id and label: # Only process valid rows
                    labels_to_encode.append(label)
                    event_id_map.append(event_id)
        
        print(f"Found {len(labels_to_encode)} valid event labels to encode.")

    except FileNotFoundError:
        print(f"--- FATAL ERROR: {EVENTS_FILE} not found. ---")
        return
    except KeyError as e:
        print(f"--- FATAL ERROR: {e} ---")
        return

    # --- Compute Embeddings ---
    print("Computing embeddings for all event labels... (This will take time)")
    start_time = time.time()
    try:
        # encode() returns a numpy array
        embeddings = model.encode(labels_to_encode, convert_to_tensor=False, show_progress_bar=True)
        print(f"Embeddings computed in {time.time() - start_time:.2f} seconds. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"--- FATAL ERROR during encoding: {e} ---")
        return

    # --- Save to JSON ---
    # Create a map of Event_ID -> Embedding (as a list)
    embeddings_map = {}
    for i, event_id in enumerate(event_id_map):
        embeddings_map[event_id] = embeddings[i].tolist() # Convert numpy array to simple list for JSON

    print(f"Saving embeddings map to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_map, f) # Save the map
        print(f"\n✅ Success! Embeddings saved to {OUTPUT_FILE}.")
        print("You can now run the Python NLP service and the Go backend.")
    except Exception as e:
        print(f"--- FATAL ERROR saving JSON: {e} ---")

if __name__ == "__main__":
    precompute()