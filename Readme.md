Karma Path: Visualizing Mahabharata Causality

Introduction

Karma Path is an interactive web application designed to explore the intricate web of cause and effect – the very paths of karma – within the epic narrative of the Mahabharata. By leveraging a comprehensive dataset mapping key events and their relationships, this tool allows users to visualize the causal chains leading to, or stemming from, significant occurrences in the story. Understand the "why" behind pivotal moments and see how events and actions cascade through the Adi, Sabha, and Vana Parvas. This project aims to provide a clear, visual understanding of the complex linkages that define this monumental work, reflecting its status as a project of utmost importance.

Features

Causal Tracing ("What led to..."): Select an event and visualize the chain of preceding events that led up to it.

Consequence Tracing ("What happened after..."): Select an event and visualize the chain of subsequent events that resulted from it.

Interactive Web Interface: Simple UI to select trace type and input event descriptions in natural language.

Semantic Event Matching: Uses sentence embeddings (via sentence-transformers) to understand the meaning behind user queries and find the most relevant event, even with variations in phrasing or minor typos.

User Disambiguation: When a query matches multiple plausible events, the application presents choices to the user, ensuring the correct event is selected before generating the flowchart.

Dynamic Flowchart Generation: Utilizes Mermaid.js to render clear, styled, and easy-to-understand flowcharts directly in the browser.

Parva Context: Flowcharts visually distinguish between events within the main story arc's Parva and preceding/subsequent "Boundary Parvas."

How It Works

Data: Event descriptions (Event_Label) with unique IDs (Event_ID) are stored in events.csv. Causal links (From_ID -> To_ID) are stored in relationships.csv.

Backend (Flask):

Loads and pre-processes the CSV data on startup.

Pre-computes sentence embeddings for all event labels using a SentenceTransformer model.

Receives user queries via API routes (/find_event_matches, /get_flowchart_by_id, /get_consequences_by_id).

Uses semantic similarity (cosine similarity on embeddings) to find candidate events matching the query (find_candidate_matches_semantic).

Performs graph traversal (backward for causes with find_all_causes, forward for consequences with find_all_consequences) based on the selected event ID.

Generates Mermaid.js syntax for the resulting event chain (generate_mermaid_code).

Returns either a list of candidate choices or the final Mermaid code to the frontend.

Frontend (HTML/JavaScript/Tailwind CSS):

Provides the user interface for input and trace type selection.

Sends the query to the Flask backend.

Displays disambiguation choices if necessary, allowing the user to select the correct event.

Sends the selected Event ID back to the appropriate Flask endpoint.

Receives the Mermaid code and renders it using the Mermaid.js library.

Screenshots

(Instructions: Replace the bracketed text below with Markdown image links. Upload your screenshots to a service like GitHub or Imgur and use the generated links.)

1. Main Interface: Shows the input field, trace type selection, and general layout.
[SCREENSHOT_MAIN_UI]
(Example: )

2. Disambiguation: Shows the application presenting multiple choices for an ambiguous query.
[SCREENSHOT_DISAMBIGUATION]
(Example: )
(You can use image_1f9af2.png here)
[Image showing the disambiguation choices presented to the user]

3. Causes Flowchart Example: A sample flowchart generated for a "What led to..." query.
[SCREENSHOT_CAUSES_FLOWCHART]
(Example: )

4. Consequences Flowchart Example: A sample flowchart generated for a "What happened after..." query.
[SCREENSHOT_CONSEQUENCES_FLOWCHART]
(Example: )

Setup and Installation

Clone the Repository:

git clone <your-repo-url>
cd <your-repo-folder-name>


Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows (Git Bash or PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


Install Dependencies:

pip install Flask sentence-transformers torch scikit-learn # Or tensorflow instead of torch


(Note: sentence-transformers will download the language model the first time the Flask app runs)

Prepare Data: Ensure events.csv and relationships.csv are present in the main project folder. They should have the following headers:

events.csv: Event_ID, Event_Label

relationships.csv: From_ID, To_ID

Run the Application:

python mahabharat_bot_flask.py


Access: Open your web browser and navigate to http://127.0.0.1:5000 (or the address provided in the terminal).

Usage

Select the desired Trace Type: "Causes" or "Consequences".

Enter a description of the event you want to trace in the input box (e.g., "What led to the game of dice?", "Consequences of Bhishma's vow?").

Click the "Find Event" button.

If multiple events match your query, click on the button corresponding to the correct event from the "Did you mean one of these?" list.

The application will display the generated flowchart tracing the causes or consequences of the selected event.

Data Source

The event mapping and relationships were derived from summaries of the Adi Parva, Sabha Parva, and Vana Parva of the Mahabharata. (You can add more detail here about the specific translation or source text used).

Future Enhancements

Expand the dataset to include all 18 Parvas.

Add filtering options (e.g., show only major events).

Implement alternative visualization methods (e.g., interactive graph networks).

Improve semantic matching with more advanced NLP models or fine-tuning.