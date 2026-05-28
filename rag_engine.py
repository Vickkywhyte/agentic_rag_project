"""
RAG Engine — 5-Step Implementation for Multi-Document Assistant
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import httpx

# ─────────────────────────────────────────────────────────────
# STEP 1: INGEST
# ─────────────────────────────────────────────────────────────

def ingest(file_bytes: bytes, filename: str) -> str:
    """Parse a file into plain text."""
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return _parse_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def _parse_pdf(file_bytes: bytes) -> str:
    import io
    for module_name in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(module_name)
            reader = mod.PdfReader(io.BytesIO(file_bytes))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            continue
        except Exception as e:
            raise ValueError(f"PDF parse error: {e}")
    raise ValueError(
        "PDF library not found. Activate your venv and run: pip install pypdf"
    )


# ─────────────────────────────────────────────────────────────
# STEP 2: CHUNK (Enhanced with keyword extraction)
# ─────────────────────────────────────────────────────────────

def chunk(text: str, chunk_size: int = 300, overlap: int = 60) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text for enhanced search."""
    # Common terms to boost
    budget_terms = [
        "budget", "professional development", "allowance", "spend", "reimbursement",
        "eligibility", "probation", "lead", "approval", "webexpenses", "expense",
        "training", "course", "conference", "coaching", "certification", "subscription",
        "book", "gbp", "usd", "eur", "aed", "aud", "bgn", "chf", "czk", "dkk", "huf",
        "isk", "jpy", "nok", "nzd", "pln", "ron", "sek", "sgd", "uah", "zak",
        "adverse media", "screening", "sanctions", "ofac", "compliance", "risk",
        "high-risk", "customer", "escalation", "sar", "financial crime"
    ]
    
    # Find terms in text
    found_terms = []
    text_lower = text.lower()
    for term in budget_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms


# ─────────────────────────────────────────────────────────────
# STEP 3: EMBED (Enhanced with TF-IDF + Keyword boost)
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """In-memory TF-IDF vector store with cosine similarity retrieval + keyword boost."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
        )
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        self.matrix = None
        self.is_fitted = False

    def add_documents(self, chunks: List[str], meta: List[Dict]):
        """Add chunks to the store with their metadata."""
        self.chunks.extend(chunks)
        self.metadata.extend(meta)
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        return len(chunks)

    def embed_query(self, query: str):
        """Convert a query to TF-IDF vector."""
        return self.vectorizer.transform([query])

    def clear(self):
        """Reset the vector store."""
        self.__init__()

    def count(self):
        """Return number of chunks stored."""
        return len(self.chunks)

    def get_all_docs(self) -> List[Dict]:
        """Get unique document sources."""
        seen, docs = set(), []
        for m in self.metadata:
            if m["source"] not in seen:
                seen.add(m["source"])
                docs.append(m)
        return docs

    def is_loaded(self):
        """Check if documents are loaded."""
        return self.count() > 0


# ─────────────────────────────────────────────────────────────
# STEP 4: RETRIEVE (Enhanced with keyword boosting)
# ─────────────────────────────────────────────────────────────

def retrieve(store: VectorStore, query: str, top_k: int = 5) -> List[Dict]:
    """Find the top-k most relevant chunks via cosine similarity + keyword boost."""
    if not store.is_fitted or store.count() == 0:
        return []

    # Get base TF-IDF similarity
    q_vec = store.embed_query(query)
    scores = cosine_similarity(q_vec, store.matrix).flatten()
    
    # Extract keywords from query for boosting
    query_keywords = set(extract_keywords(query))
    
    # Apply keyword boost to scores
    boosted_scores = scores.copy()
    for idx, chunk in enumerate(store.chunks):
        chunk_keywords = set(extract_keywords(chunk))
        # Boost score if query keywords match chunk keywords
        overlap = len(query_keywords & chunk_keywords)
        if overlap > 0:
            boosted_scores[idx] += (overlap * 0.05)  # 5% boost per matching keyword
    
    # Get top indices
    top_indices = np.argsort(boosted_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(boosted_scores[idx])
        if score > 0.01:
            results.append({
                "text": store.chunks[idx],
                "score": round(score, 4),
                "base_score": round(float(scores[idx]), 4),
                "source": store.metadata[idx]["source"],
                "chunk_index": store.metadata[idx]["chunk_index"],
                "keywords": extract_keywords(store.chunks[idx])[:5],  # Top 5 keywords
            })
    return results


# ─────────────────────────────────────────────────────────────
# STEP 5: GENERATE
# ─────────────────────────────────────────────────────────────

def generate(query: str, retrieved_chunks: List[Dict], api_key: str) -> Tuple[str, str]:
    """
    Build a grounded prompt and call the LLM via OpenRouter.
    """
    # Build context from retrieved chunks
    if not retrieved_chunks:
        context = "No relevant information found in the documents."
    else:
        parts = []
        for i, ch in enumerate(retrieved_chunks, 1):
            clean_text = ch['text'].replace('\n', ' ').strip()
            parts.append(
                f"[Source {i}: {ch['source']} | Section: {ch['chunk_index']} | Score: {ch['score']} | Keywords: {', '.join(ch['keywords'])}]\n{clean_text}"
            )
        context = "\n\n---\n\n".join(parts)

    prompt = f"""You are a helpful assistant for Wise employees. Answer the user's question using ONLY the context below which comes from official company documents.

IMPORTANT GUIDELINES:
- Be specific and accurate based on the context provided
- If the answer is not in the context, politely say you don't know
- When mentioning information, reference which document it came from

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    api_key = api_key.strip()

    # Only OpenRouter is supported
    if not api_key.startswith("sk-or-"):
        raise ValueError("This application only supports OpenRouter API keys (starting with 'sk-or-')")

    answer = _call_openrouter(prompt, api_key)

    return answer, prompt


def _call_openrouter(prompt: str, api_key: str) -> str:
    """
    Call OpenRouter using its OpenAI-compatible REST API directly.
    Includes fallback models for reliability.
    """
    # List of models to try in order
    models_to_try = [
        "anthropic/claude-3.5-sonnet",      # Best quality
        "anthropic/claude-3-haiku",          # Fast
        "mistralai/mistral-7b-instruct:free", # Free option
        "meta-llama/llama-3-8b-instruct:free", # Another free option
        "openai/gpt-3.5-turbo",               # Fallback
    ]
    
    last_error = None
    
    for model in models_to_try:
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Wise Multi-Document Assistant",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                last_error = f"Model {model} failed: {response.status_code}"
                continue
                
        except Exception as e:
            last_error = f"Model {model} error: {str(e)}"
            continue
    
    raise ValueError(f"All OpenRouter models failed. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────
# DOCUMENT LOADING FUNCTION (for any document)
# ─────────────────────────────────────────────────────────────

def load_document(store: VectorStore, doc_path: str):
    """Load any document into the vector store with its filename as source."""
    try:
        print(f"📖 Loading document from {doc_path}...")
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excess newlines
        
        # Chunk the document
        chunks = chunk(text, chunk_size=300, overlap=60)
        
        # Get just the filename (not the full path)
        source_name = os.path.basename(doc_path)
        
        # Create metadata with the ACTUAL filename as source
        meta = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
        
        # Add to store
        store.add_documents(chunks, meta)
        
        print(f"✅ Loaded {len(chunks)} chunks from {source_name}")
        print(f"📊 Total chunks in store: {store.count()}")
        
        # Print sample keywords from first few chunks
        print(f"\n🔑 Sample keywords from first 3 chunks of {source_name}:")
        for i in range(min(3, len(chunks))):
            keywords = extract_keywords(chunks[i])
            print(f"  Chunk {i}: {', '.join(keywords[:5])}")
        
        return True
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {doc_path}")
        return False
    except Exception as e:
        print(f"❌ Error loading {doc_path}: {e}")
        return False
