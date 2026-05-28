"""
Agentic RAG Application - FastAPI Backend with Agentic Mode
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

from rag_engine import VectorStore, load_document
from src.agent_executor import AgenticRAGExecutor

# ── App setup ────────────────────────────────────────────────

app = FastAPI(title="Wise Agentic RAG Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global stores
store = VectorStore()
agentic_executor = None

# ── Config ───────────────────────────────────────────────────

TOP_K = 5
MODE = "agentic"  # Can be "standard" or "agentic"

# ── Load Documents at startup ────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load documents and initialize agentic executor"""
    print("🚀 Starting Wise Agentic RAG Assistant...")
    print(f"📝 Mode: {MODE.upper()}")
    
    documents = ["Development_Budget.txt", "adverse_media.txt"]
    
    loaded_count = 0
    for doc in documents:
        if os.path.exists(doc):
            success = load_document(store, doc)
            if success:
                loaded_count += 1
        else:
            print(f"⚠️ Warning: {doc} not found in current directory")
    
    # Initialize agentic executor if in agentic mode
    global agentic_executor
    if MODE == "agentic":
        agentic_executor = AgenticRAGExecutor(store)
        print("🤖 Agentic executor initialized")
    
    total = store.count()
    if total > 0:
        print(f"✅ Loaded {loaded_count} documents with {total} chunks")
        print(f"🌐 Open http://localhost:8000")
    else:
        print("❌ No documents loaded. Make sure Development_Budget.txt and adverse_media.txt exist")

# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the HTML interface with API key input"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Wise Agentic RAG Assistant</title>
    <style>
        body { 
            font-family: system-ui, -apple-system, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #0a0c10; 
            color: #e5e7eb; 
        }
        .header { 
            text-align: center; 
            padding: 40px; 
            background: linear-gradient(135deg, #1e1b4b, #0c4a6e); 
            border-radius: 16px; 
            margin-bottom: 30px; 
        }
        .mode-badge { 
            display: inline-block; 
            background: #10b981; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-size: 12px; 
            font-weight: bold; 
        }
        .api-section { 
            background: #14161c; 
            border-radius: 16px; 
            padding: 20px; 
            margin-bottom: 20px; 
            border: 1px solid #2a2d36; 
        }
        .api-input { 
            display: flex; 
            gap: 12px; 
            align-items: center; 
            margin-top: 12px; 
        }
        .api-input input { 
            flex: 1; 
            background: #0a0c10; 
            border: 1px solid #2a2d36; 
            border-radius: 8px; 
            padding: 12px; 
            color: white; 
            font-family: monospace; 
        }
        .api-input input:focus { 
            outline: none; 
            border-color: #10b981; 
        }
        .api-input button { 
            background: #10b981; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 8px; 
            color: white; 
            font-weight: bold; 
            cursor: pointer; 
        }
        .api-input button:hover { 
            background: #059669; 
        }
        .api-status { 
            font-size: 12px; 
            margin-top: 8px; 
            color: #6b7280; 
        }
        .api-status a { 
            color: #10b981; 
        }
        .chat-container { 
            background: #14161c; 
            border-radius: 16px; 
            padding: 20px; 
            border: 1px solid #2a2d36; 
            min-height: 500px; 
            display: flex;
            flex-direction: column;
        }
        .messages { 
            flex: 1; 
            min-height: 400px; 
            max-height: 500px; 
            overflow-y: auto; 
            margin-bottom: 20px;
        }
        .message { 
            margin-bottom: 16px; 
            padding: 12px; 
            border-radius: 8px; 
        }
        .user { 
            background: #10b981; 
            text-align: right; 
        }
        .assistant { 
            background: #1a1d24; 
            border-left: 3px solid #10b981; 
        }
        .trace { 
            font-size: 11px; 
            color: #6b7280; 
            margin-top: 8px; 
            padding: 8px; 
            background: #0a0c10; 
            border-radius: 4px; 
        }
        .trace details { 
            cursor: pointer; 
        }
        .trace pre { 
            margin-top: 8px; 
            overflow-x: auto; 
        }
        .input-area { 
            display: flex; 
            gap: 12px; 
            margin-top: 20px; 
        }
        textarea { 
            flex: 1; 
            background: #0a0c10; 
            border: 1px solid #2a2d36; 
            border-radius: 8px; 
            padding: 12px; 
            color: white; 
            font-family: monospace; 
            resize: vertical;
        }
        textarea:focus { 
            outline: none; 
            border-color: #10b981; 
        }
        button { 
            background: #10b981; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 8px; 
            color: white; 
            font-weight: bold; 
            cursor: pointer; 
        }
        button:hover:not(:disabled) { 
            background: #059669; 
        }
        button:disabled { 
            opacity: 0.5; 
            cursor: not-allowed; 
        }
        .quick-questions { 
            display: flex; 
            gap: 8px; 
            flex-wrap: wrap; 
            margin-top: 20px; 
        }
        .pill { 
            background: #1a1d24; 
            border: 1px solid #2a2d36; 
            padding: 6px 12px; 
            border-radius: 20px; 
            font-size: 12px; 
            cursor: pointer; 
            transition: all 0.2s;
        }
        .pill:hover { 
            border-color: #10b981; 
            background: rgba(16, 185, 129, 0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-ready { background: #10b981; box-shadow: 0 0 5px #10b981; }
        .status-waiting { background: #f59e0b; }
        .status-error { background: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Wise Agentic RAG Assistant</h1>
        <div class="mode-badge">⚡ AGENTIC MODE</div>
        <p>Multi-step reasoning · Tool use · Verification</p>
    </div>
    
    <div class="api-section">
        <strong>🔑 OpenRouter API Key Required</strong>
        <div class="api-input">
            <input type="password" id="apiKeyInput" placeholder="sk-or-..." value="">
            <button onclick="saveApiKey()">Save Key</button>
        </div>
        <div class="api-status" id="apiStatus">
            <span class="status-indicator status-waiting"></span>
            No API key set. Get a free key at <a href="https://openrouter.ai" target="_blank">openrouter.ai</a>
        </div>
    </div>
    
    <div class="chat-container">
        <div class="messages" id="messages">
            <div class="message assistant">
                <strong>Assistant:</strong><br>
                👋 Hello! I'm an Agentic RAG system. I can:
                <ul style="margin-top: 8px; margin-bottom: 8px;">
                    <li>🔍 Search documents</li>
                    <li>🧮 Calculate numbers</li>
                    <li>✅ Verify information across sources</li>
                    <li>🔁 Execute multi-step plans</li>
                </ul>
                <strong>➡️ First, enter your OpenRouter API key above, then ask me anything about Wise policies!</strong>
            </div>
        </div>
        
        <div class="quick-questions">
            <div class="pill" onclick="setQuery('How much is the development budget for Product team?')">💰 Budget amount</div>
            <div class="pill" onclick="setQuery('Compare the budget for Product vs Finance teams')">📊 Compare budgets</div>
            <div class="pill" onclick="setQuery('What can I spend my budget on? Then calculate 20% of the budget')">🧮 Calculate percentage</div>
            <div class="pill" onclick="setQuery('Find what the adverse media policy says about sanctions screening')">✅ Find policy</div>
            <div class="pill" onclick="setQuery('First find eligibility criteria for the budget, then find the reimbursement process')">🔁 Multi-step</div>
            <div class="pill" onclick="setQuery('Can I use my budget for a conference? What about ChatGPT subscription?')">🎫 Conference & subscriptions</div>
        </div>
        
        <div class="input-area">
            <textarea id="query" rows="2" placeholder="Ask a complex question that requires planning..."></textarea>
            <button id="sendBtn" onclick="sendQuery()" disabled>Send</button>
        </div>
    </div>
    
    <script>
        // Load saved API key from localStorage
        let apiKey = localStorage.getItem('openrouter_api_key') || '';
        if (apiKey) {
            document.getElementById('apiKeyInput').value = apiKey;
            updateApiStatus(true);
            document.getElementById('sendBtn').disabled = false;
        }
        
        function updateApiStatus(isValid) {
            const statusDiv = document.getElementById('apiStatus');
            if (isValid) {
                statusDiv.innerHTML = '<span class="status-indicator status-ready"></span>✅ API key saved and ready! You can now ask questions.';
            } else {
                statusDiv.innerHTML = '<span class="status-indicator status-error"></span>❌ Invalid API key. Must start with "sk-or-". Get a key at <a href="https://openrouter.ai" target="_blank">openrouter.ai</a>';
            }
        }
        
        function saveApiKey() {
            const newKey = document.getElementById('apiKeyInput').value.trim();
            if (newKey && newKey.startsWith('sk-or-')) {
                apiKey = newKey;
                localStorage.setItem('openrouter_api_key', apiKey);
                updateApiStatus(true);
                document.getElementById('sendBtn').disabled = false;
                addMessage('assistant', '✅ API key saved successfully! You can now ask questions about Wise policies.', false);
            } else {
                updateApiStatus(false);
                document.getElementById('sendBtn').disabled = true;
                if (newKey && !newKey.startsWith('sk-or-')) {
                    addMessage('assistant', '❌ Invalid API key format. OpenRouter keys must start with "sk-or-". Please check your key.', false);
                }
            }
        }
        
        async function sendQuery() {
            const queryTextarea = document.getElementById('query');
            const query = queryTextarea.value.trim();
            if (!query) return;
            
            // Check API key
            if (!apiKey || !apiKey.startsWith('sk-or-')) {
                addMessage('assistant', '❌ Please enter a valid OpenRouter API key first (starts with sk-or-). Get one at https://openrouter.ai', false);
                return;
            }
            
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.textContent = 'Thinking...';
            
            // Add user message and clear input
            addMessage('user', query, true);
            queryTextarea.value = '';
            queryTextarea.style.height = 'auto';
            
            // Add thinking indicator
            const thinkingId = addThinkingMessage();
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, api_key: apiKey })
                });
                
                const data = await response.json();
                
                // Remove thinking indicator
                removeThinkingMessage(thinkingId);
                
                if (response.ok) {
                    // Format the answer with trace information
                    let answerHtml = data.answer;
                    if (data.execution_trace && data.execution_trace.tools_used && data.execution_trace.tools_used.length > 0) {
                        answerHtml += `
                            <div class="trace">
                                <details>
                                    <summary>🔍 Agentic Trace (Tools used: ${data.execution_trace.tools_used.join(', ')})</summary>
                                    <pre style="font-size: 10px; overflow-x: auto; margin-top: 8px;">${JSON.stringify(data.execution_trace.plan, null, 2)}</pre>
                                </details>
                            </div>
                        `;
                    } else if (data.retrieved_chunks) {
                        answerHtml += `
                            <div class="trace">
                                <details>
                                    <summary>🔍 Standard RAG Trace (${data.retrieved_chunks.length} chunks retrieved)</summary>
                                    <div style="font-size: 10px; margin-top: 8px;">
                                        ${data.retrieved_chunks.map((c, i) => `<div style="margin-bottom: 4px;">📄 Source ${i+1}: ${c.source} (score: ${c.score})</div>`).join('')}
                                    </div>
                                </details>
                            </div>
                        `;
                    }
                    addMessage('assistant', answerHtml, false);
                } else {
                    addMessage('assistant', '❌ Error: ' + (data.detail || 'Unknown error'), false);
                }
            } catch (err) {
                removeThinkingMessage(thinkingId);
                addMessage('assistant', '❌ Network error: ' + err.message + '. Make sure the server is running.', false);
            }
            
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        }
        
        function addMessage(role, content, isUser) {
            const messagesDiv = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            if (isUser) {
                div.innerHTML = `<strong>You:</strong><br>${escapeHtml(content)}`;
            } else {
                div.innerHTML = `<strong>Assistant:</strong><br>${content}`;
            }
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return div;
        }
        
        function addThinkingMessage() {
            const messagesDiv = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message assistant';
            const id = 'thinking-' + Date.now();
            div.id = id;
            div.innerHTML = '<strong>Assistant:</strong><br><em>🤔 Thinking, planning, and executing steps...</em>';
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return id;
        }
        
        function removeThinkingMessage(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }
        
        function setQuery(q) {
            document.getElementById('query').value = q;
            // Auto-resize textarea
            const textarea = document.getElementById('query');
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
            sendQuery();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Auto-resize textarea as user types
        document.getElementById('query').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        // Enter to send (Shift+Enter for new line)
        document.getElementById('query').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
            }
        });
    </script>
</body>
</html>
    """


class QueryRequest(BaseModel):
    query: str
    api_key: str = ""


@app.post("/query")
async def query_documents(req: QueryRequest):
    """Execute query using agentic workflow"""
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents loaded. Please contact administrator.")
    
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key required (starts with 'sk-or-').")
    
    if not api_key.startswith("sk-or-"):
        raise HTTPException(status_code=400, detail="Invalid API key format. Must start with 'sk-or-'")
    
    try:
        if MODE == "agentic" and agentic_executor:
            # Use Agentic RAG
            result = agentic_executor.execute(req.query, api_key)
            return {
                "answer": result["answer"],
                "execution_trace": result["execution_trace"]
            }
        else:
            # Fallback to standard RAG
            from rag_engine import retrieve, generate
            results = retrieve(store, req.query, top_k=TOP_K)
            answer, prompt = generate(req.query, results, api_key)
            return {
                "answer": answer,
                "retrieved_chunks": results,
                "execution_trace": {"mode": "standard", "tools_used": ["rag_search"]}
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Get system status"""
    return {
        "mode": MODE,
        "chunk_count": store.count(),
        "documents_loaded": store.count() > 0,
        "agentic_ready": agentic_executor is not None
    }