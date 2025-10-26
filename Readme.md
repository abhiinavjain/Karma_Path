# Karma Path: Visualizing Mahabharata Causality

## Introduction

Karma Path is an interactive web application designed to explore the intricate web of cause and effect – the very paths of karma – within the epic narrative of the Mahabharata. By leveraging a comprehensive dataset mapping key events and their relationships, this tool allows users to visualize the causal chains leading to, or stemming from, significant occurrences in the story. Understand the "why" behind pivotal moments and see how events and actions cascade through the Adi, Sabha, and Vana Parvas. This project aims to provide a clear, visual understanding of the complex linkages that define this monumental work, reflecting its status as a project of utmost importance.

## Images

### Consequences 
#### Consequenses of Bhima meeting Hanuman ? 
<img width="1002" height="607" alt="image" src="https://github.com/user-attachments/assets/9992147a-7757-448d-a98a-35f133374ee7" />

### Cause 
#### Question: What led to Arjuna winning Draupadi at the Swayamvar? 
<img width="511" height="603" alt="Cause" src="https://github.com/user-attachments/assets/9642e4a6-addd-4980-8b1f-5e8ab5c1f4c9" />



## Video Overview 

https://github.com/user-attachments/assets/bdf67288-c9a7-4545-9402-bb26266883aa

## Features

* **Causal Tracing ("What led to..."):** Select an event and visualize the chain of preceding events that led up to it.
* **Consequence Tracing ("What happened after..."):** Select an event and visualize the chain of subsequent events that resulted from it.
* **Interactive Web Interface:** Simple UI to select trace type and input event descriptions in natural language.
* **Semantic Event Matching:** Uses sentence embeddings (via `sentence-transformers`) to understand the meaning behind user queries and find the most relevant event, even with variations in phrasing or minor typos.
* **User Disambiguation:** When a query matches multiple plausible events, the application presents choices to the user, ensuring the correct event is selected before generating the flowchart.
* **Dynamic Flowchart Generation:** Utilizes Mermaid.js to render clear, styled, and easy-to-understand flowcharts directly in the browser.
* **Parva Context:** Flowcharts visually distinguish between events within the main story arc's Parva and preceding/subsequent "Boundary Parvas."

## How It Works

1.  **Data:** Event descriptions (`Event_Label`) with unique IDs (`Event_ID`) are stored in `events.csv`. Causal links (`From_ID` -> `To_ID`) are stored in `relationships.csv`.
2.  **Backend (Flask):**
    * Loads and pre-processes the CSV data on startup.
    * Pre-computes sentence embeddings for all event labels using a `SentenceTransformer` model.
    * Receives user queries via API routes (`/find_event_matches`, `/get_flowchart_by_id`, `/get_consequences_by_id`).
    * Uses semantic similarity (cosine similarity on embeddings) to find candidate events matching the query (`find_candidate_matches_semantic`).
    * Performs graph traversal (backward for causes with `find_all_causes`, forward for consequences with `find_all_consequences`) based on the selected event ID.
    * Generates Mermaid.js syntax for the resulting event chain (`generate_mermaid_code`).
    * Returns either a list of candidate choices or the final Mermaid code to the frontend.
3.  **Frontend (HTML/JavaScript/Tailwind CSS):**
    * Provides the user interface for input and trace type selection.
    * Sends the query to the Flask backend.
    * Displays disambiguation choices if necessary, allowing the user to select the correct event.
    * Sends the selected Event ID back to the appropriate Flask endpoint.
    * Receives the Mermaid code and renders it using the Mermaid.js library.


## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/abhiinavjain/Karma_Path>
    cd <Karma_Path>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows (Git Bash or PowerShell):
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install Flask sentence-transformers torch scikit-learn # Or tensorflow instead of torch
    ```
    *(Note: `sentence-transformers` will download the language model the first time the Flask app runs)*
4.  **Prepare Data:** Ensure `events.csv` and `relationships.csv` are present in the main project folder. They should have the following headers:
    * `events.csv`: `Event_ID`, `Event_Label`
    * `relationships.csv`: `From_ID`, `To_ID`
5.  **Run the Application:**
    ```bash
    python mahabharat_bot_flask.py
    ```
6.  **Access:** Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided in the terminal).

## Usage

1.  Select the desired **Trace Type**: "Causes" or "Consequences".
2.  Enter a description of the event you want to trace in the input box (e.g., "What led to the game of dice?", "Consequences of Bhishma's vow?").
3.  Click the "Find Event" button.
4.  If multiple events match your query, click on the button corresponding to the correct event from the "Did you mean one of these?" list.
5.  The application will display the generated flowchart tracing the causes or consequences of the selected event.

## Data Source

The event mapping and relationships were derived from summaries of the Adi Parva, Sabha Parva, and Vana Parva of the Mahabharata. 

## Future Enhancements

* Expand the dataset to include all 18 Parvas.
* Add filtering options (e.g., show only major events).
* Implement alternative visualization methods (e.g., interactive graph networks).
* Improve semantic matching with more advanced NLP models or fine-tuning.
