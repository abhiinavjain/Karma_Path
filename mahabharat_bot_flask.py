import csv
import os
import sys
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

app = Flask(__name__)

all_nodes = []
all_links = []
embedding_model = None
event_label_embeddings = None 
event_labels_list = []      
node_list_for_semantic = [] 

# It gives the parva of which the event is sab62- > sab 
def get_prefix(event_id):
    prefix = ""; event_id_str = str(event_id)
    for char in event_id_str:
        if char.isalpha(): prefix += char
        else: break
    return prefix

# It opens events.csv and relationships.csv, reads them row by row, and populates the global all_nodes and all_links lists.
def load_data_and_embeddings():
    """Loads CSV data AND pre-computes sentence embeddings for labels."""
    global all_nodes, all_links, embedding_model, event_label_embeddings, event_labels_list, node_list_for_semantic
    nodes = []; links = []
    print("Loading knowledge base...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        events_path = os.path.join(script_dir, 'events.csv')
        relationships_path = os.path.join(script_dir, 'relationships.csv')
        if not os.path.exists(events_path): raise FileNotFoundError(f"Cannot find 'events.csv' in '{script_dir}'")
        if not os.path.exists(relationships_path): raise FileNotFoundError(f"Cannot find 'relationships.csv' in '{script_dir}'")

        # Load Nodes
        temp_nodes_for_embedding = []
        with open(events_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if 'Event_ID' not in reader.fieldnames or 'Event_Label' not in reader.fieldnames: raise KeyError("events.csv headers missing.")
            # Read all nodes first
            all_loaded_nodes = list(reader)
            # Filter out nodes without ID or Label *before* embedding
            nodes = [node for node in all_loaded_nodes if node.get('Event_ID') and node.get('Event_Label')]
            temp_nodes_for_embedding = [node for node in nodes] # Keep a list in order for embedding index mapping

        # Load Links
        with open(relationships_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if 'From_ID' not in reader.fieldnames or 'To_ID' not in reader.fieldnames: raise KeyError("relationships.csv headers missing.")
            links = [{'from': row['From_ID'], 'to': row['To_ID']} for row in reader if row.get('From_ID') and row.get('To_ID')]

        all_nodes = nodes # Store the filtered nodes globally
        all_links = links
        node_list_for_semantic = temp_nodes_for_embedding # Store the ordered list globally
        event_labels_list = [node['Event_Label'] for node in node_list_for_semantic] # Store labels globally
        print(f"Successfully loaded {len(all_nodes)} valid events and {len(all_links)} links.")

        # --- Initialize Sentence Transformer Model ---
        # Using a relatively lightweight but effective model.
        # It will be downloaded automatically the first time.
        model_name = 'all-MiniLM-L6-v2'
        print(f"Loading sentence transformer model '{model_name}'... (This may take a while on first run)")
        embedding_model = SentenceTransformer(model_name)
        print("Model loaded.")

        # --- Pre-compute Embeddings for Event Labels ---
        print("Pre-computing embeddings for all event labels...")
        # Encode the list of labels
        # Convert to numpy for easier handling with scikit-learn or direct tensor ops
        event_label_embeddings = embedding_model.encode(event_labels_list, convert_to_tensor=False, show_progress_bar=True)
        # Or convert to torch tensor if using pytorch directly for similarity:
        # event_label_embeddings = embedding_model.encode(event_labels_list, convert_to_tensor=True, show_progress_bar=True)
        print(f"Embeddings computed. Shape: {event_label_embeddings.shape}")

        return True
    except Exception as e: print(f"--- FATAL ERROR loading data/embeddings: {e} ---"); import traceback; traceback.print_exc(); return False

# Searches the all_nodes list for a specific Event_ID and returns that event's full dictionary (e.g., {'Event_ID': 'sab62', 'Event_Label': '...'}).
def get_event_from_id(event_id):
    """Finds and returns the full event dictionary using its ID."""
    
    return next((node for node in all_nodes if node.get('Event_ID') == event_id), None)

# --- REVISED Matching Function using Semantic Similarity ---
def find_candidate_matches_semantic(query, threshold=40, limit=5):
    """
    Finds potential event matches using Sentence Embedding Cosine Similarity.
    Threshold is 0-100 scale (converted from cosine similarity 0-1).
    Returns a list of dictionaries: [{'id': Event_ID, 'label': Event_Label, 'score': score}]
    """
    global embedding_model, event_label_embeddings, node_list_for_semantic

    if embedding_model is None or event_label_embeddings is None or not node_list_for_semantic:
        print("[ERROR] Semantic search model not initialized or no embeddings available.")
        return []

    try:
        # Encode the user query using the same model
        query_embedding = embedding_model.encode([query], convert_to_tensor=False) # Get numpy array

        # Calculate cosine similarity between query and all pre-computed label embeddings
        # Using scikit-learn's cosine_similarity for numpy arrays
        # Input shape: (1, embed_dim), (num_labels, embed_dim) -> Output: (1, num_labels)
        cosine_scores = cosine_similarity(query_embedding, event_label_embeddings)[0] # Get the 1D array of scores

        # Convert scores to 0-100 scale
        scores_100 = (cosine_scores * 100).round().astype(int)

        # Get indices of scores above the threshold
        relevant_indices = np.where(scores_100 >= threshold)[0]

        # Create candidate list with scores, mapping index back to node
        candidates_with_scores = []
        for i in relevant_indices:
            score = scores_100[i]
            # Map index back to the original node data using the ordered list
            if i < len(node_list_for_semantic):
                 node = node_list_for_semantic[i]
                 candidates_with_scores.append({
                     'id': node.get('Event_ID'),
                     'label': node.get('Event_Label'),
                     'score': int(score) # Ensure score is int
                 })
            else:
                 print(f"[Warning] Index {i} out of bounds for node_list_for_semantic.")


        # Sort candidates by score (descending) and limit
        candidates_with_scores.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates_with_scores[:limit]

        print(f"    Found {len(candidates)} semantic candidates with score >= {threshold}.")
        return candidates

    except Exception as e:
        print(f"--- ERROR during semantic matching: {e} ---")
        import traceback; traceback.print_exc()
        return []


# --- Core "Causes Finder" Logic (Unchanged) ---
def find_all_causes(target_event_id, target_prefix):
    # ... (Keep the working version from previous steps) ...
    events_in_chain = {}; links_in_chain = []
    events_to_process = [(target_event_id, target_prefix)]; events_queued = {target_event_id}
    target_node_details = get_event_from_id(target_event_id)
    if target_node_details:
        node_copy = target_node_details.copy(); node_copy['is_boundary'] = False
        events_in_chain[target_event_id] = node_copy
    max_loops = len(all_nodes) * 3; loops = 0; processed_nodes = set()
    while events_to_process and loops < max_loops:
        current_event_id, current_arc_prefix = events_to_process.pop(0); loops += 1
        if current_event_id in processed_nodes: continue
        processed_nodes.add(current_event_id)
        for link in all_links:
            if link.get('to') == current_event_id:
                cause_event_id = link.get('from')
                if not cause_event_id: continue
                link_tuple = (cause_event_id, current_event_id)
                if link_tuple not in [(l.get('from'), l.get('to')) for l in links_in_chain]: links_in_chain.append(link.copy())
                cause_prefix = get_prefix(cause_event_id)
                if cause_event_id not in events_in_chain:
                    event_details = get_event_from_id(cause_event_id)
                    if event_details:
                        node_copy = event_details.copy(); node_copy['is_boundary'] = (cause_prefix != target_prefix)
                        events_in_chain[cause_event_id] = node_copy
                elif cause_prefix != target_prefix:
                     if events_in_chain[cause_event_id].get('is_boundary') is None: events_in_chain[cause_event_id]['is_boundary'] = True
                should_trace_further = (cause_prefix == current_arc_prefix)
                if should_trace_further and cause_event_id not in events_queued:
                     events_queued.add(cause_event_id); events_to_process.append((cause_event_id, current_arc_prefix))
                elif cause_prefix != current_arc_prefix and cause_event_id not in events_queued:
                     events_queued.add(cause_event_id); events_to_process.append((cause_event_id, cause_prefix))
    if loops >= max_loops: print(f"\n[Warning: Max trace depth reached for {target_event_id}. Causes flowchart might be truncated.]")
    final_node_ids = set(events_in_chain.keys())
    for link in links_in_chain:
        for node_id_key in ['from', 'to']:
            node_id = link.get(node_id_key)
            if node_id and node_id not in final_node_ids:
                 node_details = get_event_from_id(node_id)
                 if node_details:
                      node_copy = node_details.copy(); node_copy['is_boundary'] = (get_prefix(node_id) != target_prefix)
                      events_in_chain[node_id] = node_copy; final_node_ids.add(node_id)
    return list(events_in_chain.values()), links_in_chain

# --- Core "Consequences Finder" Logic (Unchanged) ---
def find_all_consequences(start_event_id, start_prefix):
    # ... (Keep the working version from previous steps) ...
    events_in_chain = {}; links_in_chain = []
    events_to_process = [(start_event_id, start_prefix)]; events_queued = {start_event_id}
    start_node_details = get_event_from_id(start_event_id)
    if start_node_details:
        node_copy = start_node_details.copy(); node_copy['is_boundary'] = False
        events_in_chain[start_event_id] = node_copy
    max_loops = len(all_nodes) * 3; loops = 0; processed_nodes = set()
    while events_to_process and loops < max_loops:
        current_event_id, current_arc_prefix = events_to_process.pop(0); loops += 1
        if current_event_id in processed_nodes: continue
        processed_nodes.add(current_event_id)
        for link in all_links:
            if link.get('from') == current_event_id:
                consequence_event_id = link.get('to')
                if not consequence_event_id: continue
                link_tuple = (current_event_id, consequence_event_id)
                if link_tuple not in [(l.get('from'), l.get('to')) for l in links_in_chain]: links_in_chain.append(link.copy())
                consequence_prefix = get_prefix(consequence_event_id)
                if consequence_event_id not in events_in_chain:
                    event_details = get_event_from_id(consequence_event_id)
                    if event_details:
                        node_copy = event_details.copy(); node_copy['is_boundary'] = (consequence_prefix != start_prefix)
                        events_in_chain[consequence_event_id] = node_copy
                elif consequence_prefix != start_prefix:
                     if events_in_chain[consequence_event_id].get('is_boundary') is None: events_in_chain[consequence_event_id]['is_boundary'] = True
                should_trace_further = (consequence_prefix == current_arc_prefix)
                if should_trace_further and consequence_event_id not in events_queued:
                     events_queued.add(consequence_event_id); events_to_process.append((consequence_event_id, current_arc_prefix))
                elif consequence_prefix != current_arc_prefix and consequence_event_id not in events_queued:
                     events_queued.add(consequence_event_id); events_to_process.append((consequence_event_id, consequence_prefix))
    if loops >= max_loops: print(f"\n[Warning: Max trace depth reached for {start_event_id}. Consequences flowchart might be truncated.]")
    final_node_ids = set(events_in_chain.keys())
    for link in links_in_chain:
        for node_id_key in ['from', 'to']:
            node_id = link.get(node_id_key)
            if node_id and node_id not in final_node_ids:
                 node_details = get_event_from_id(node_id)
                 if node_details:
                      node_copy = node_details.copy(); node_copy['is_boundary'] = (get_prefix(node_id) != start_prefix)
                      events_in_chain[node_id] = node_copy; final_node_ids.add(node_id)
    return list(events_in_chain.values()), links_in_chain

# --- Flowchart Code Generator (Unchanged) ---
# --- REVISED Flowchart Code Generator (Simplified Nodes, Safer Labels) ---
def generate_mermaid_code(start_or_target_event_id, start_or_target_label, nodes_in_chain, links_in_chain, trace_type="causes"):
    """
    Generates Mermaid flowchart code using safer label handling.
    """
    direction = "TD"
    code = f"graph {direction};\n"

    # Define CSS classes (Keep these, they are applied separately)
    code += "    classDef start fill:#6366f1,stroke:#4338ca,stroke-width:3px,color:#ffffff,font-weight:bold,padding:10px;\n"
    code += "    classDef cause fill:#f1f5f9,stroke:#94a3b8,stroke-width:2px,color:#1e293b,padding:8px;\n"
    code += "    classDef consequence fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#1e293b,padding:8px;\n"
    code += "    classDef boundary fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#713f12,padding:8px,stroke-dasharray: 5 5;\n"

    code += "\n    %% --- The Events ---\n"
    processed_ids = set()
    for node in nodes_in_chain:
        event_id = node.get('Event_ID')
        label = node.get('Event_Label', f'Label missing {event_id}')
        is_boundary = node.get('is_boundary', False)

        if not event_id or event_id in processed_ids: continue
        processed_ids.add(event_id)

        # --- Safer Label Sanitization ---
        # 1. Replace backticks with alternatives (like single quotes) as backticks are used for quoting.
        # 2. Escape double quotes that might be inside the label if using double quote delimiters.
        #    Using backticks `` ` `` is generally safer if the text might contain quotes.
        # label_text_sanitized = label.replace('`', "'").replace('(', '[').replace(')', ']')
        # Alternative: Escape double quotes if using " " delimiters
        label_text_sanitized = label.replace('"', '#quot;').replace('(', '[').replace(')', ']') # Use HTML entity for quotes

        # Add boundary note directly
        if is_boundary and event_id != start_or_target_event_id:
             label_text_sanitized += " (Boundary Parva)" # Simple text addition

        # --- Simplified Node Definition ---
        node_definition = ""
        node_class_application = ""
        node_shape_start = "[" ; node_shape_end = "]" # Default rectangle
        node_class_name = "cause" # Default

        if event_id == start_or_target_event_id:
            node_class_name = "start"
            node_shape_start = "((" if trace_type == "causes" else "(["
            node_shape_end = "))" if trace_type == "causes" else "])"
        elif trace_type == "consequences":
            node_class_name = "consequence"
        # Boundary style overrides only if applicable
        if is_boundary and event_id != start_or_target_event_id:
             node_class_name = "boundary"

        # Use Mermaid's double-quote string literal syntax for the label
        node_definition = f'    {event_id}{node_shape_start}"{label_text_sanitized}"{node_shape_end}'
        node_class_application = f'    class {event_id} {node_class_name};'
        # --- End Simplified Node Definition ---


        code += node_definition + "\n"
        code += node_class_application + "\n"
        code += f'    click {event_id} call handleNodeClick("{event_id}")\n'


    code += "\n    %% --- The Connections ---\n"
    valid_node_ids = processed_ids # Use the set of nodes actually defined
    added_links = set()
    for link in links_in_chain:
         from_id = link.get('from'); to_id = link.get('to')
         link_tuple = (from_id, to_id)
         # Ensure both linked nodes were successfully defined above
         if from_id in valid_node_ids and to_id in valid_node_ids and link_tuple not in added_links:
              code += f"    {from_id} --> {to_id}\n"; added_links.add(link_tuple)
         elif link_tuple not in added_links: # Avoid multiple warnings
             print(f"[Warning/MermaidGen] Skipping link {from_id}-->{to_id}; one or both nodes missing/skipped.")
             added_links.add(link_tuple)

    code += "\n    %% --- Link Styles ---\n"
    code += "    linkStyle default stroke:#6b7280,stroke-width:2px;\n"

    # --- Add Final Debug Print ---
    print("\n--- Generated Mermaid Code ---")
    print(code)
    print("-----------------------------\n")
    # --- End Debug Print ---

    return code

# --- (The rest of your code remains the same) ---
# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/family-tree')
def family_tree():
    return render_template('family-tree.html')

# --- REVISED Route for initial query using TF-IDF ---
@app.route('/find_event_matches', methods=['POST'])
def find_event_matches():
    """
    Receives user query, finds candidate matches using Semantic Similarity (TF-IDF),
    and returns them OR proceeds directly if only one very strong match.
    """
    try:
        data = request.get_json(); user_question = data.get('question', '').strip()
        trace_type = data.get('trace_type', 'causes')
        print("\n" + "="*30); print(f"[DEBUG] Received MATCH query: '{user_question}' (Type: {trace_type})")
        if not user_question: return jsonify({'error': 'No question provided.'}), 400

        # Use the TF-IDF candidate finder
        CANDIDATE_THRESHOLD = 30 # Lower threshold for semantic similarity (0-100 scale)
        DIRECT_THRESHOLD = 75    # Threshold to proceed directly (0-100 scale)
        print(f"[DEBUG] Finding semantic candidates with threshold >= {CANDIDATE_THRESHOLD}...")
        # find_candidate_matches_semantic expects threshold 0-100
        candidates = find_candidate_matches_semantic(user_question, threshold=CANDIDATE_THRESHOLD, limit=5)

        if not candidates:
            print(f" --> No semantic candidates found above threshold {CANDIDATE_THRESHOLD}.")
            return jsonify({'error': f"Could not find any relevant matches (Score < {CANDIDATE_THRESHOLD}). Please try rephrasing with key names or terms."}), 404

        # Check if the top match is strong enough to proceed directly
        proceed_directly = False
        if len(candidates) == 1 and candidates[0]['score'] >= DIRECT_THRESHOLD:
             proceed_directly = True
        elif len(candidates) > 1 and candidates[0]['score'] >= DIRECT_THRESHOLD:
             # Only proceed if top score is significantly higher than the next (e.g., > 15 points)
             if (candidates[0]['score'] - candidates[1]['score']) > 15:
                  proceed_directly = True

        if proceed_directly:
            print(f" --> Found 1 strong semantic match (Score: {candidates[0]['score']} >= {DIRECT_THRESHOLD}). Proceeding directly...")
            target_event_id = candidates[0]['id']
            target_label = candidates[0]['label']
            target_prefix = get_prefix(target_event_id)

            if trace_type == "consequences":
                 print(f"[DEBUG] Calling find_all_consequences for ID: '{target_event_id}'")
                 chain_nodes, chain_links = find_all_consequences(target_event_id, target_prefix)
            else:
                 print(f"[DEBUG] Calling find_all_causes for ID: '{target_event_id}'")
                 chain_nodes, chain_links = find_all_causes(target_event_id, target_prefix)

            if not chain_nodes: return jsonify({'error': f'Failed to trace {trace_type}.'}), 500

            print(f"[DEBUG] Calling generate_mermaid_code (trace_type='{trace_type}')...")
            mermaid_code_string = generate_mermaid_code(target_event_id, target_label, chain_nodes, chain_links, trace_type=trace_type)
            print(" --> Generated Mermaid code successfully.")
            print("="*30)
            return jsonify({
                'mermaid_code': mermaid_code_string, 'target_label': target_label,
                'user_question': user_question, 'trace_type': trace_type,
                'status': 'success'
            })
        else: # Multiple candidates or top score not high enough - return choices
            print(f" --> Found {len(candidates)} plausible semantic matches (Top score: {candidates[0]['score']}). Returning choices.")
            print("="*30)
            return jsonify({
                'choices': candidates, 'user_question': user_question,
                'trace_type': trace_type, 'status': 'disambiguation_required'
            })

    except Exception as e:
        print(f"--- ERROR processing /find_event_matches request: {e} ---"); import traceback; traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred during matching.'}), 500


# --- Route for generating flowchart from a specific ID (Unchanged) ---
@app.route('/get_flowchart_by_id', methods=['POST'])
def get_flowchart_by_id():
    """ Receives a specific Event ID and trace type, generates flowchart, returns JSON. """
    try:
        data = request.get_json()
        target_event_id = data.get('event_id'); trace_type = data.get('trace_type', 'causes')
        user_question = data.get('user_question', 'N/A')
        print("\n" + "="*30); print(f"[DEBUG] Received ID query: '{target_event_id}' (Type: {trace_type})")
        if not target_event_id: return jsonify({'error': 'No Event ID provided.'}), 400

        target_node = get_event_from_id(target_event_id)
        if not target_node:
            print(f" --> ERROR: Node not found for ID '{target_event_id}'.")
            return jsonify({'error': f"Event ID '{target_event_id}' not found."}), 404

        target_label = target_node.get('Event_Label', 'Label Missing')
        target_prefix = get_prefix(target_event_id)
        print(f" --> Generating flowchart for: [{target_event_id}] {target_label}")

        if trace_type == "consequences":
             print(f"[DEBUG] Calling find_all_consequences for ID: '{target_event_id}'")
             chain_nodes, chain_links = find_all_consequences(target_event_id, target_prefix)
        else:
             print(f"[DEBUG] Calling find_all_causes for ID: '{target_event_id}'")
             chain_nodes, chain_links = find_all_causes(target_event_id, target_prefix)

        if not chain_nodes: return jsonify({'error': f'Failed to trace {trace_type} from ID.'}), 500

        print(f"[DEBUG] Calling generate_mermaid_code (trace_type='{trace_type}')...")
        mermaid_code_string = generate_mermaid_code(target_event_id, target_label, chain_nodes, chain_links, trace_type=trace_type)
        print(" --> Generated Mermaid code successfully.")
        print("="*30)
        return jsonify({
            'mermaid_code': mermaid_code_string, 'target_label': target_label,
            'user_question': user_question, 'trace_type': trace_type,
            'status': 'success'
        })

    except Exception as e:
        print(f"--- ERROR processing /get_flowchart_by_id request: {e} ---"); import traceback; traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred generating flowchart by ID.'}), 500
# --- NEW Route for generating CONSEQUENCES flowchart from a specific ID ---
@app.route('/get_consequences_by_id', methods=['POST'])
def get_consequences_by_id():
    """
    Receives a specific Event ID and trace type ('consequences'),
    generates flowchart tracing FORWARD, returns JSON. Bypasses matching.
    """
    try:
        data = request.get_json()
        target_event_id = data.get('event_id')
        trace_type = data.get('trace_type', 'consequences') # Default to consequences
        user_question = data.get('user_question', 'N/A')

        print("\n" + "="*30); print(f"[DEBUG] Received CONSEQUENCE ID query: '{target_event_id}' (Type: {trace_type})")

        if not target_event_id: return jsonify({'error': 'No Event ID provided.'}), 400

        target_node = get_event_from_id(target_event_id)
        if not target_node:
            print(f" --> ERROR: Node not found for ID '{target_event_id}'.")
            return jsonify({'error': f"Event ID '{target_event_id}' not found."}), 404

        target_label = target_node.get('Event_Label', 'Label Missing')
        target_prefix = get_prefix(target_event_id) # Start prefix
        print(f" --> Generating consequences flowchart STARTING FROM: [{target_event_id}] {target_label}")

        # --- Call find_all_consequences ---
        print(f"[DEBUG] Calling find_all_consequences for ID: '{target_event_id}'")
        chain_nodes, chain_links = find_all_consequences(target_event_id, target_prefix)
        print(f"[DEBUG] find_all_consequences returned {len(chain_nodes)} nodes and {len(chain_links)} links.")
        # --- End Call ---

        if not chain_nodes: return jsonify({'error': f'Failed to trace consequences from ID.'}), 500

        print(f"[DEBUG] Calling generate_mermaid_code (trace_type='consequences')...")
        mermaid_code_string = generate_mermaid_code(target_event_id, target_label, chain_nodes, chain_links, trace_type="consequences") # Explicitly set trace_type
        print(" --> Generated Mermaid code successfully.")
        print("="*30)

        return jsonify({
            'mermaid_code': mermaid_code_string, 'target_label': target_label,
            'user_question': user_question, 'trace_type': trace_type,
            'status': 'success'
        })

    except Exception as e:
        print(f"--- ERROR processing /get_consequences_by_id request: {e} ---"); import traceback; traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred generating consequences flowchart by ID.'}), 500

# --- (Keep the main run block `if __name__ == '__main__':` the same) ---
# --- Run the Flask App ---
if __name__ == '__main__':
    if not load_data_and_embeddings(): # Use the new loading function
        print("\n--- Exiting due to data loading failure. ---"); sys.exit(1)

    print("\n--- Starting Flask Development Server ---")
    print(" * Data loaded successfully.")
    print(" * Embeddings computed and model ready.") # Updated message
    print(" * Open your web browser and go to: http://127.0.0.1:5000 (or your local IP:5000)")
    print(" * Press CTRL+C in this terminal to stop the server.")
    app.run(debug=True, host='0.0.0.0', port=5000)