package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	// "time" // Not used
)

// --- Structs ---

// EventNode holds data from events.csv
type EventNode struct {
	EventID    string `json:"id"`    // Used for JSON response
	Label      string `json:"label"` // Used for JSON response
	IsBoundary bool   `json:"-"`     // Internal use, ignore in JSON
}

// EventLink holds data from relationships.csv
type EventLink struct {
	From string
	To   string
}

// Candidate holds a potential match for disambiguation
type Candidate struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Score int    `json:"score"`
}

// --- Globals ---
var (
	allNodes        map[string]EventNode // O(1) lookup by Event_ID
	causesMap       map[string][]string  // Adjacency list for causes
	consequencesMap map[string][]string  // Adjacency list for consequences
	eventEmbeddings map[string][]float64 // Pre-computed embeddings
	templates       *template.Template   // HTML templates

	// URL for the Python NLP service
	nlpServiceURL = os.Getenv("nlp_url")
)

// --- Helper: Get Parva prefix ---
func getPrefix(eventID string) string {
	for i, r := range eventID {
		if r < 'a' || r > 'z' {
			return eventID[:i]
		}
	}
	return eventID
}

// --- Data Loading ---
func loadData() error {
	log.Println("Loading knowledge base...")

	// Initialize maps
	allNodes = make(map[string]EventNode)
	causesMap = make(map[string][]string)
	consequencesMap = make(map[string][]string)
	eventEmbeddings = make(map[string][]float64)

	// Get base directory
	scriptDir, err := os.Getwd() // Get Current Working Directory
	if err != nil {
		return fmt.Errorf("could not find script directory: %v", err)
	}

	// 1. Load Nodes (events.csv)
	eventsPath := filepath.Join(scriptDir, "events.csv")
	file, err := os.Open(eventsPath)
	if err != nil {
		return fmt.Errorf("cannot find 'events.csv' in '%s': %v", scriptDir, err)
	}
	defer file.Close()

	// Use utf-8-sig logic by trimming the BOM
	bom := []byte{0xEF, 0xBB, 0xBF}
	buffer := make([]byte, 3)
	n, _ := file.Read(buffer)
	if n != 3 || !bytes.Equal(buffer, bom) {
		file.Seek(0, 0) // Not a BOM or read error, rewind
	}

	r := csv.NewReader(file)
	r.LazyQuotes = true
	header, err := r.Read()
	if err != nil {
		return fmt.Errorf("failed to read events.csv header: %v", err)
	}

	// Find column indices
	idIdx, labelIdx := -1, -1
	for i, col := range header {
		if col == "Event_ID" {
			idIdx = i
		}
		if col == "Event_Label" {
			labelIdx = i
		}
	}
	if idIdx == -1 || labelIdx == -1 {
		return fmt.Errorf("events.csv must have 'Event_ID' and 'Event_Label' headers")
	}

	records, err := r.ReadAll()
	if err != nil {
		return fmt.Errorf("failed to read events.csv records: %v", err)
	}

	for _, rec := range records {
		if len(rec) > idIdx && len(rec) > labelIdx && rec[idIdx] != "" && rec[labelIdx] != "" {
			allNodes[rec[idIdx]] = EventNode{EventID: rec[idIdx], Label: rec[labelIdx]}
		}
	}
	log.Printf("Successfully loaded %d valid events.", len(allNodes))

	// 2. Load Links (relationships.csv) & Build Adjacency Maps
	linksPath := filepath.Join(scriptDir, "relationships.csv")
	fileLinks, err := os.Open(linksPath)
	if err != nil {
		return fmt.Errorf("cannot find 'relationships.csv' in '%s': %v", scriptDir, err)
	}
	defer fileLinks.Close()

	// Reset buffer for BOM check
	n, _ = fileLinks.Read(buffer)
	if n != 3 || !bytes.Equal(buffer, bom) {
		fileLinks.Seek(0, 0)
	}

	rLinks := csv.NewReader(fileLinks)
	rLinks.LazyQuotes = true
	headerLinks, err := rLinks.Read()
	if err != nil {
		return fmt.Errorf("failed to read relationships.csv header: %v", err)
	}

	fromIdx, toIdx := -1, -1
	for i, col := range headerLinks {
		if col == "From_ID" {
			fromIdx = i
		}
		if col == "To_ID" {
			toIdx = i
		}
	}
	if fromIdx == -1 || toIdx == -1 {
		return fmt.Errorf("relationships.csv must have 'From_ID' and 'To_ID' headers")
	}

	recordsLinks, err := rLinks.ReadAll()
	if err != nil {
		return fmt.Errorf("failed to read relationships.csv records: %v", err)
	}

	linkCount := 0
	for _, rec := range recordsLinks {
		if len(rec) > fromIdx && len(rec) > toIdx && rec[fromIdx] != "" && rec[toIdx] != "" {
			fromID, toID := rec[fromIdx], rec[toIdx]
			// Build maps ONLY if nodes exist (data integrity)
			if _, ok := allNodes[fromID]; ok {
				if _, ok := allNodes[toID]; ok {
					causesMap[toID] = append(causesMap[toID], fromID)
					consequencesMap[fromID] = append(consequencesMap[fromID], toID)
					linkCount++
				}
			}
		}
	}
	log.Printf("Successfully loaded %d valid links and built adjacency maps.", linkCount)

	// 3. Load Pre-computed Embeddings (event_embeddings.json)
	embedPath := filepath.Join(scriptDir, "event_embeddings.json")
	fileEmbed, err := os.Open(embedPath)
	if err != nil {
		return fmt.Errorf("cannot find 'event_embeddings.json' in '%s': %v", scriptDir, err)
	}
	defer fileEmbed.Close()

	if err = json.NewDecoder(fileEmbed).Decode(&eventEmbeddings); err != nil {
		return fmt.Errorf("failed to parse event_embeddings.json: %v", err)
	}
	log.Printf("Successfully loaded %d pre-computed embeddings.", len(eventEmbeddings))

	// 4. Load HTML Templates
	templates, err = template.ParseGlob(filepath.Join(scriptDir, "templates", "*.html"))
	if err != nil {
		return fmt.Errorf("failed to parse HTML templates in 'templates' folder: %v", err)
	}
	log.Println("HTML templates loaded.")

	return nil
}

// --- Helper: Get Event from ID (O(1) lookup) ---
func getEventFromID(eventID string) (EventNode, bool) {
	node, ok := allNodes[eventID]
	return node, ok
}

// --- Semantic Matching Logic ---

// encodeQuery calls the Python NLP microservice
func encodeQuery(query string) ([]float64, error) {
	requestBody, err := json.Marshal(map[string]string{"query": query})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal query: %v", err)
	}

	resp, err := http.Post(nlpServiceURL, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NLP service at %s: %v", nlpServiceURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("NLP service returned error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string][]float64
	if err = json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse NLP service response: %v", err)
	}

	vector, ok := result["vector"]
	if !ok {
		return nil, fmt.Errorf("NLP service response missing 'vector' key")
	}
	return vector, nil
}

// dotProduct calculates A · B
func dotProduct(v1, v2 []float64) (float64, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("vectors have different lengths")
	}
	var dot float64
	for i := range v1 {
		dot += v1[i] * v2[i]
	}
	return dot, nil
}

// magnitude calculates ||V||
func magnitude(v []float64) float64 {
	var sumOfSquares float64
	for _, val := range v {
		sumOfSquares += val * val
	}
	return math.Sqrt(sumOfSquares)
}

// cosineSimilarity calculates (A · B) / (||A|| * ||B||)
func cosineSimilarity(v1, v2 []float64) (float64, error) {
	dot, err := dotProduct(v1, v2)
	if err != nil {
		return 0, err
	}
	mag1 := magnitude(v1)
	mag2 := magnitude(v2)
	if mag1 == 0 || mag2 == 0 {
		return 0, nil // Avoid division by zero
	}
	return dot / (mag1 * mag2), nil
}

func findCandidateMatchesSemantic(query string, threshold int, limit int) ([]Candidate, error) {
	log.Printf("Finding semantic candidates with threshold >= %d...", threshold)

	queryVector, err := encodeQuery(query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %v", err)
	}

	var candidates []Candidate

	for eventID, eventVector := range eventEmbeddings {
		sim, err := cosineSimilarity(queryVector, eventVector)
		if err != nil {
			log.Printf("[Warning] Failed to compare with %s: %v", eventID, err)
			continue
		}

		score := int(math.Round(sim * 100))
		if score >= threshold {
			if node, ok := allNodes[eventID]; ok {
				candidates = append(candidates, Candidate{
					ID:    eventID,
					Label: node.Label,
					Score: score,
				})
			}
		}
	}

	// Sort by score (descending)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	log.Printf("    Found %d semantic candidates with score >= %d.", len(candidates), threshold)
	return candidates, nil
}

// --- Graph Traversal Logic (BFS) ---
// (Note: find_all_causes and find_all_consequences are implemented below, outside the handlers)

// --- Mermaid Code Generator ---
// ################## THIS IS THE FIX ##################
// Added 'targetLabel string' as the second argument
func generateMermaidCode(startOrTargetEventID string, targetLabel string, nodesInChain map[string]EventNode, linksInChain []EventLink, traceType string) string {
	// #####################################################
	var code strings.Builder
	direction := "TD"
	if traceType == "consequences" {
		direction = "LR"
	}
	code.WriteString(fmt.Sprintf("graph %s;\n", direction))

	// Define styles
	code.WriteString("    classDef start fill:#6366f1,stroke:#4338ca,stroke-width:3px,color:#ffffff,font-weight:bold,padding:10px;\n")
	code.WriteString("    classDef cause fill:#f1f5f9,stroke:#94a3b8,stroke-width:2px,color:#1e293b,padding:8px;\n")
	code.WriteString("    classDef consequence fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#1e293b,padding:8px;\n")
	code.WriteString("    classDef boundary fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#713f12,padding:8px,stroke-dasharray: 5 5;\n")

	code.WriteString("\n    %% --- The Events ---\n")
	processedIDs := make(map[string]bool)

	// Iterate over the map of nodes
	for eventID, node := range nodesInChain {
		if processedIDs[eventID] {
			continue
		}
		processedIDs[eventID] = true

		// Sanitize label (basic)
		label := node.Label
		label = strings.ReplaceAll(label, "\"", "#quot;")
		label = strings.ReplaceAll(label, "(", "[")
		label = strings.ReplaceAll(label, ")", "]")

		// Word wrapping (simple version)
		maxLineLength := 50
		if eventID == startOrTargetEventID {
			maxLineLength = 40
		}
		if len(label) > maxLineLength {
			// Basic word wrapping logic can be added here if needed
		}

		nodeDefinition, nodeClass := "", ""
		nodeShapeStart, nodeShapeEnd := "[", "]"
		nodeClassName := "cause"

		if eventID == startOrTargetEventID {
			nodeClassName = "start"
			if traceType == "causes" {
				nodeShapeStart, nodeShapeEnd = "((", "))"
			} else {
				nodeShapeStart, nodeShapeEnd = "([", "])"
			}
		} else if traceType == "consequences" {
			nodeClassName = "consequence"
		}

		if node.IsBoundary && eventID != startOrTargetEventID {
			nodeClassName = "boundary"
			label += "<br/><span style='font-style:italic; font-size: 0.8em;'>(Boundary Parva)</span>"
		}

		nodeDefinition = fmt.Sprintf(`    %s%s"<div style='padding:5px;'>%s</div>"%s`, eventID, nodeShapeStart, label, nodeShapeEnd)
		nodeClass = fmt.Sprintf(`    class %s %s;`, eventID, nodeClassName)

		code.WriteString(nodeDefinition + "\n")
		code.WriteString(nodeClass + "\n")
	}

	code.WriteString("\n    %% --- The Connections ---\n")
	addedLinks := make(map[string]bool)
	for _, link := range linksInChain {
		linkKey := link.From + "->" + link.To
		if _, ok := processedIDs[link.From]; ok {
			if _, ok := processedIDs[link.To]; ok {
				if !addedLinks[linkKey] {
					code.WriteString(fmt.Sprintf("    %s --> %s\n", link.From, link.To))
					addedLinks[linkKey] = true
				}
			}
		}
	}

	code.WriteString("\n    %% --- Link Styles ---\n")
	code.WriteString("    linkStyle default stroke:#6b7280,stroke-width:2px;\n")

	// log.Println("--- Generated Mermaid Code ---") // Optional debug
	// log.Println(code.String())
	// log.Println("-----------------------------")

	return code.String()
}

// --- HTTP Handlers ---

func indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	err := templates.ExecuteTemplate(w, "index.html", nil)
	if err != nil {
		log.Printf("[ERROR] Failed to render index.html: %v", err)
		http.Error(w, "Internal Server Error", 500)
	}
}

func familyTreeHandler(w http.ResponseWriter, r *http.Request) {
	err := templates.ExecuteTemplate(w, "family-tree.html", nil)
	if err != nil {
		log.Printf("[ERROR] Failed to render family-tree.html: %v", err)
		http.Error(w, "Internal Server Error", 500)
	}
}

// Struct for decoding JSON requests
type MatchRequest struct {
	Question  string `json:"question"`
	TraceType string `json:"trace_type"`
}
type IDRequest struct {
	EventID      string `json:"event_id"`
	TraceType    string `json:"trace_type"`
	UserQuestion string `json:"user_question"`
}

// Struct for sending JSON responses
type MatchResponse struct {
	Choices      []Candidate `json:"choices,omitempty"`
	UserQuestion string      `json:"user_question"`
	TraceType    string      `json:"trace_type"`
	Status       string      `json:"status"` // "disambiguation_required" or "success"
	MermaidCode  string      `json:"mermaid_code,omitempty"`
	TargetLabel  string      `json:"target_label,omitempty"`
	Error        string      `json:"error,omitempty"`
}

func findEventMatchesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST allowed", 405)
		return
	}

	var req MatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400)
		return
	}

	log.Printf("\n==============================\n[DEBUG] Received MATCH query: '%s' (Type: %s)", req.Question, req.TraceType)
	if req.Question == "" {
		http.Error(w, `{"error": "No question provided."}`, 400)
		return
	}

	candidateThreshold := 45 // 0-100 scale
	directThreshold := 75    // 0-100 scale
	log.Printf("[DEBUG] Finding semantic candidates with threshold >= %d...", candidateThreshold)

	candidates, err := findCandidateMatchesSemantic(req.Question, candidateThreshold, 5)
	if err != nil {
		log.Printf(" --> [ERROR] in findCandidateMatchesSemantic: %v", err)
		http.Error(w, `{"error": "Failed to run semantic match."}`, 500)
		return
	}

	if len(candidates) == 0 {
		log.Printf(" --> No semantic candidates found above threshold %d.", candidateThreshold)
		http.Error(w, `{"error": "Could not find any relevant matches."}`, 404)
		return
	}

	proceedDirectly := false
	if len(candidates) == 1 && candidates[0].Score >= directThreshold {
		proceedDirectly = true
	} else if len(candidates) > 1 && candidates[0].Score >= directThreshold && (candidates[0].Score-candidates[1].Score) > 15 {
		proceedDirectly = true
	}

	var resp MatchResponse

	if proceedDirectly {
		log.Printf(" --> Found 1 strong semantic match (Score: %d >= %d). Proceeding directly...", candidates[0].Score, directThreshold)
		targetEventID := candidates[0].ID
		targetLabel := candidates[0].Label
		targetPrefix := getPrefix(targetEventID)

		var chainNodes []EventNode
		var chainLinks []EventLink

		if req.TraceType == "consequences" {
			log.Printf("[DEBUG] Calling find_all_consequences for ID: '%s'", targetEventID)
			chainNodes, chainLinks = findAllConsequences(targetEventID, targetPrefix)
		} else {
			log.Printf("[DEBUG] Calling find_all_causes for ID: '%s'", targetEventID)
			chainNodes, chainLinks = findAllCauses(targetEventID, targetPrefix)
		}

		if len(chainNodes) == 0 {
			http.Error(w, `{"error": "Failed to trace."}`, 500)
			return
		}

		// --- THIS CALL IS NOW CORRECT (5 arguments) ---
		mermaidCode := generateMermaidCode(targetEventID, targetLabel, chainNodesToMap(chainNodes), chainLinks, req.TraceType)
		log.Println(" --> Generated Mermaid code successfully.")

		resp = MatchResponse{
			UserQuestion: req.Question,
			TraceType:    req.TraceType,
			Status:       "success",
			MermaidCode:  mermaidCode,
			TargetLabel:  targetLabel,
		}

	} else {
		log.Printf(" --> Found %d plausible semantic matches (Top score: %d). Returning choices.", len(candidates), candidates[0].Score)
		resp = MatchResponse{
			Choices:      candidates,
			UserQuestion: req.Question,
			TraceType:    req.TraceType,
			Status:       "disambiguation_required",
		}
	}

	log.Println("==============================")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// --- Helper for graph traversal ---
func chainNodesToMap(nodes []EventNode) map[string]EventNode {
	m := make(map[string]EventNode)
	for _, node := range nodes {
		m[node.EventID] = node
	}
	return m
}

// --- BFS Traversal Functions (Go Version) ---

func getFlowchartByIDHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST allowed", 405)
		return
	}

	var req IDRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400)
		return
	}

	log.Printf("\n==============================\n[DEBUG] Received ID query: '%s' (Type: %s)", req.EventID, req.TraceType)
	if req.EventID == "" {
		http.Error(w, `{"error": "No Event ID provided."}`, 400)
		return
	}

	targetNode, ok := getEventFromID(req.EventID)
	if !ok {
		log.Printf(" --> ERROR: Node not found for ID '%s'.", req.EventID)
		http.Error(w, `{"error": "Event ID not found."}`, 404)
		return
	}

	targetLabel := targetNode.Label
	targetPrefix := getPrefix(targetNode.EventID)
	log.Printf(" --> Generating flowchart for: [%s] %s", req.EventID, targetLabel)

	// --- find_all_causes logic, Go version ---
	chainNodes, chainLinks := findAllCauses(req.EventID, targetPrefix)

	if len(chainNodes) == 0 {
		http.Error(w, `{"error": "Failed to trace causes from ID."}`, 500)
		return
	}

	log.Printf("[DEBUG] Calling generate_mermaid_code (trace_type='causes')...")
	// --- THIS CALL IS NOW CORRECT (5 arguments) ---
	mermaidCode := generateMermaidCode(req.EventID, targetLabel, chainNodesToMap(chainNodes), chainLinks, "causes")
	log.Println(" --> Generated Mermaid code successfully.")
	log.Println("==============================")

	resp := MatchResponse{
		UserQuestion: req.UserQuestion,
		TraceType:    req.TraceType,
		Status:       "success",
		MermaidCode:  mermaidCode,
		TargetLabel:  targetLabel,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func getConsequencesByIDHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST allowed", 405)
		return
	}

	var req IDRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", 400)
		return
	}

	log.Printf("\n==============================\n[DEBUG] Received CONSEQUENCE ID query: '%s' (Type: %s)", req.EventID, req.TraceType)
	if req.EventID == "" {
		http.Error(w, `{"error": "No Event ID provided."}`, 400)
		return
	}

	targetNode, ok := getEventFromID(req.EventID)
	if !ok {
		log.Printf(" --> ERROR: Node not found for ID '%s'.", req.EventID)
		http.Error(w, `{"error": "Event ID not found."}`, 404)
		return
	}

	targetLabel := targetNode.Label
	targetPrefix := getPrefix(targetNode.EventID)
	log.Printf(" --> Generating consequences flowchart STARTING FROM: [%s] %s", req.EventID, targetLabel)

	// --- find_all_consequences logic, Go version ---
	chainNodes, chainLinks := findAllConsequences(req.EventID, targetPrefix)

	if len(chainNodes) == 0 {
		http.Error(w, `{"error": "Failed to trace consequences from ID."}`, 500)
		return
	}

	log.Printf("[DEBUG] Calling generate_mermaid_code (trace_type='consequences')...")
	// --- THIS CALL IS NOW CORRECT (5 arguments) ---
	mermaidCode := generateMermaidCode(req.EventID, targetLabel, chainNodesToMap(chainNodes), chainLinks, "consequences")
	log.Println(" --> Generated Mermaid code successfully.")
	log.Println("==============================")

	resp := MatchResponse{
		UserQuestion: req.UserQuestion,
		TraceType:    req.TraceType,
		Status:       "success",
		MermaidCode:  mermaidCode,
		TargetLabel:  targetLabel,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// --- BFS Traversal Functions (Go Version) ---
// (These are the Go translations of your Python logic)

func findAllCauses(targetEventID string, targetPrefix string) ([]EventNode, []EventLink) {
	eventsInChain := make(map[string]EventNode)
	var linksInChain []EventLink
	queue := []struct{ id, prefix string }{{targetEventID, targetPrefix}}
	eventsQueued := map[string]bool{targetEventID: true}

	targetNode, ok := getEventFromID(targetEventID)
	if ok {
		targetNode.IsBoundary = false
		eventsInChain[targetEventID] = targetNode
	}

	maxLoops := len(allNodes) * 3
	loops := 0
	processedNodes := make(map[string]bool)

	for len(queue) > 0 && loops < maxLoops {
		current := queue[0]
		queue = queue[1:]
		loops++
		if processedNodes[current.id] {
			continue
		}
		processedNodes[current.id] = true

		// Use the pre-computed adjacency map
		if causes, ok := causesMap[current.id]; ok {
			for _, causeEventID := range causes {
				linkTuple := causeEventID + "->" + current.id
				if _, ok := findLink(linksInChain, linkTuple); !ok {
					linksInChain = append(linksInChain, EventLink{From: causeEventID, To: current.id})
				}

				causePrefix := getPrefix(causeEventID)
				if _, ok := eventsInChain[causeEventID]; !ok {
					eventDetails, ok := getEventFromID(causeEventID)
					if ok {
						eventDetails.IsBoundary = (causePrefix != targetPrefix)
						eventsInChain[causeEventID] = eventDetails
					}
				} else if causePrefix != targetPrefix {
					if node, ok := eventsInChain[causeEventID]; ok && !node.IsBoundary {
						// This check is slightly different, but aims for same goal
					} else {
						node.IsBoundary = true
						eventsInChain[causeEventID] = node
					}
				}

				shouldTraceFurther := (causePrefix == current.prefix)
				if shouldTraceFurther && !eventsQueued[causeEventID] {
					eventsQueued[causeEventID] = true
					queue = append(queue, struct{ id, prefix string }{causeEventID, current.prefix})
				} else if causePrefix != current.prefix && !eventsQueued[causeEventID] {
					eventsQueued[causeEventID] = true
					queue = append(queue, struct{ id, prefix string }{causeEventID, causePrefix})
				}
			}
		}
	}
	if loops >= maxLoops {
		log.Printf("[Warning] Max trace depth reached for %s", targetEventID)
	}

	// Convert map to slice
	var nodes []EventNode
	for _, node := range eventsInChain {
		nodes = append(nodes, node)
	}
	return nodes, linksInChain
}

func findAllConsequences(startEventID string, startPrefix string) ([]EventNode, []EventLink) {
	eventsInChain := make(map[string]EventNode)
	var linksInChain []EventLink
	queue := []struct{ id, prefix string }{{startEventID, startPrefix}}
	eventsQueued := map[string]bool{startEventID: true}

	startNode, ok := getEventFromID(startEventID)
	if ok {
		startNode.IsBoundary = false
		eventsInChain[startEventID] = startNode
	}

	maxLoops := len(allNodes) * 3
	loops := 0
	processedNodes := make(map[string]bool)

	for len(queue) > 0 && loops < maxLoops {
		current := queue[0]
		queue = queue[1:]
		loops++
		if processedNodes[current.id] {
			continue
		}
		processedNodes[current.id] = true

		// Use the pre-computed adjacency map
		if consequences, ok := consequencesMap[current.id]; ok {
			for _, consequenceEventID := range consequences {
				linkTuple := current.id + "->" + consequenceEventID
				if _, ok := findLink(linksInChain, linkTuple); !ok {
					linksInChain = append(linksInChain, EventLink{From: current.id, To: consequenceEventID})
				}

				consequencePrefix := getPrefix(consequenceEventID)
				if _, ok := eventsInChain[consequenceEventID]; !ok {
					eventDetails, ok := getEventFromID(consequenceEventID)
					if ok {
						eventDetails.IsBoundary = (consequencePrefix != startPrefix)
						eventsInChain[consequenceEventID] = eventDetails
					}
				} else if consequencePrefix != startPrefix {
					if node, ok := eventsInChain[consequenceEventID]; ok && !node.IsBoundary {
						//
					} else {
						node.IsBoundary = true
						eventsInChain[consequenceEventID] = node
					}
				}

				shouldTraceFurther := (consequencePrefix == current.prefix)
				if shouldTraceFurther && !eventsQueued[consequenceEventID] {
					eventsQueued[consequenceEventID] = true
					queue = append(queue, struct{ id, prefix string }{consequenceEventID, current.prefix})
				} else if consequencePrefix != current.prefix && !eventsQueued[consequenceEventID] {
					eventsQueued[consequenceEventID] = true
					queue = append(queue, struct{ id, prefix string }{consequenceEventID, consequencePrefix})
				}
			}
		}
	}
	if loops >= maxLoops {
		log.Printf("[Warning] Max trace depth reached for %s", startEventID)
	}

	var nodes []EventNode
	for _, node := range eventsInChain {
		nodes = append(nodes, node)
	}
	return nodes, linksInChain
}

// Helper to check for link duplicates
func findLink(links []EventLink, key string) (EventLink, bool) {
	for _, link := range links {
		if (link.From + "->" + link.To) == key {
			return link, true
		}
	}
	return EventLink{}, false
}

// --- Main Function ---
func main() {
	// Load all data into globals at startup
	if err := loadData(); err != nil {
		log.Fatalf("--- Exiting due to data loading failure: %v ---", err)
	}

	if nlpServiceURL == "" {
		log.Println("[Warning] NLP_SERVICE_URL env variable not set. Defaulting to local http://127.0.0.1:5001/encode")
		nlpServiceURL = "http://127.0.0.1:5001/encode"
	}

	// Define HTTP routes
	http.HandleFunc("/", indexHandler)
	http.HandleFunc("/family-tree", familyTreeHandler)
	http.HandleFunc("/find_event_matches", findEventMatchesHandler)
	http.HandleFunc("/get_flowchart_by_id", getFlowchartByIDHandler)
	http.HandleFunc("/get_consequences_by_id", getConsequencesByIDHandler)

	// Start the server
	port := "5000"
	log.Println("\n--- Starting Go HTTP Server ---")
	log.Println(" * Data loaded successfully.")
	log.Println(" * Embeddings loaded.")
	log.Printf(" * Open your web browser and go to: http://127.0.0.1:%s", port)
	log.Println(" * (Ensure the Python NLP service is also running on port 5001)")
	log.Println(" * Press CTRL+C in this terminal to stop the server.")

	err := http.ListenAndServe("0.0.0.0:"+port, nil)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
