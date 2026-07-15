import streamlit as st
import ollama
import os
import re
import math
from collections import Counter

st.set_page_config(page_title="Alenso's BOT",layout="centered")
st.title("Alenso's myBOT Demo")
st.write("This is a simple demo of how Alenso will take a user query, retrieve relevant chunks of information from a vector database, and then use those chunks as context to generate a response from a language model.")

# Model Configurations
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


def split_sentences(paragraph):
    """Splits a paragraph into sentence-level chunks on '.', '?', '!'."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]


# --- 1. Vector DB Setup & Caching ---
@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    dataset_path = 'alenso.txt'
    
    # Check if file exists to prevent crashing
    if not os.path.exists(dataset_path):
        # Creating a fallback file for demonstration if it doesn't exist
        with open(dataset_path, 'w') as f:
            f.write("Alenso (Alen Alex) is a premium AI/ML Engineer, Founder, Product Strategist, and Business Scaling Expert based in Thiruvananthapuram, Kerala, India.\n\n")
            f.write("Alenso is the founder of ALENSO CREATION, a multi-divisional corporate structure spanning SORGIN (AI automation and GTM consulting), ZINTH (SaaS products like HostFlow), and an R&D department focused on AI patents.\n\n")
            f.write("Alenso's premium services include AI automation and system design, RAG pipelines, Go-To-Market strategy, SaaS architecture, and strategic branding and product management.\n")

    with open(dataset_path, 'r', encoding="utf-8") as file:
        raw_text = file.read()

    # Split into paragraph-level chunks (separated by blank lines) so each
    # embedding covers a complete thought instead of a fragment of one, then
    # also split each paragraph into sentence-level chunks. Having both
    # granularities in the index means broad questions match the paragraph
    # and single-fact questions (e.g. "does he do karate?") match the exact
    # sentence, which a single granularity would often miss or dilute.
    paragraphs = re.split(r'\n\s*\n', raw_text)

    dataset = []
    seen = set()
    for paragraph in paragraphs:
        cleaned = ' '.join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        dataset.append(cleaned)

        for sentence in split_sentences(cleaned):
            # Skip bare headers/fragments too short to carry standalone meaning.
            if len(sentence.split()) < 4 or sentence in seen:
                continue
            seen.add(sentence)
            dataset.append(sentence)

    vector_db = []

    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i, chunk in enumerate(dataset):
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return []
        progress_bar.progress((i + 1) / len(dataset))

    # Clear the loading indicators when done
    status_text.empty()
    progress_bar.empty()
    return vector_db

# Common filler words dropped before BM25 scoring, so lexical matching is
# driven by content words (e.g. "karate") rather than conversational noise
# (e.g. "i", "that", "is") that would otherwise dilute the ranking.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "he", "she", "it", "they", "you", "we", "that", "this", "these",
    "those", "of", "in", "on", "at", "to", "for", "with", "and", "or",
    "but", "do", "does", "did", "so", "if", "then", "than", "as", "by",
    "from", "about", "into", "up", "down", "just", "can", "will", "would",
    "could", "should", "his", "her", "its", "their", "my", "your", "our",
}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def build_bm25_index(chunks):
    """Builds a lightweight BM25 (Okapi) index over the given chunks - the
    word-by-word lexical counterpart to the vector semantic embeddings."""
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_chunks]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

    doc_freq = Counter()
    for tokens in tokenized_chunks:
        doc_freq.update(set(tokens))

    num_docs = len(tokenized_chunks)
    idf = {
        term: math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    return {
        "tokenized_chunks": tokenized_chunks,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length or 1,
        "idf": idf,
    }


def bm25_scores(query, bm25_index, k1=1.5, b=0.75):
    """Scores every chunk in the BM25 index against the query's terms."""
    query_terms = tokenize(query)
    scores = []
    for tokens, doc_length in zip(bm25_index["tokenized_chunks"], bm25_index["doc_lengths"]):
        term_freq = Counter(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            idf = bm25_index["idf"].get(term, 0)
            denom = freq + k1 * (1 - b + b * doc_length / bm25_index["avg_doc_length"])
            score += idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


# Initialize the database
with st.spinner("Initializing Vector Database and generating embeddings..."):
    VECTOR_DB = initialize_vector_db()

BM25_INDEX = build_bm25_index([chunk for chunk, _ in VECTOR_DB])


# --- 2. Helper Functions ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if int(norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)

## Hybrid retrieval: fuses vector semantic search (meaning-based) with BM25
## lexical search (exact keyword-based) via Reciprocal Rank Fusion, so a
## query is well served whether the match is semantic or a literal keyword.
def retrieve(query, top_n=4, rrf_k=60):
    chunks = [chunk for chunk, _ in VECTOR_DB]

    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    vector_ranking = sorted(
        range(len(VECTOR_DB)),
        key=lambda i: cosine_similarity(query_embedding, VECTOR_DB[i][1]),
        reverse=True,
    )

    lexical_scores = bm25_scores(query, BM25_INDEX)
    bm25_ranking = sorted(range(len(chunks)), key=lambda i: lexical_scores[i], reverse=True)

    # Reciprocal Rank Fusion: combine the two rankings by position rather than
    # raw score, since cosine similarity (0-1) and BM25 (unbounded) aren't on
    # comparable scales.
    fused_scores = {}
    for rank, idx in enumerate(vector_ranking):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_ranking):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

    top_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]
    return [(chunks[i], fused_scores[i]) for i in top_indices]


# --- 3. UI and Interaction ---

# Sidebar to show the retrieved context chunks behind the scenes
with st.sidebar:
    st.header("Database Info")
    st.success(f"Loaded {len(VECTOR_DB)} items from dataset.")
    st.markdown("---")
    st.subheader("Retrieved Context Window")
    context_placeholder = st.empty()
    context_placeholder.info("Ask a question to see the matching context chunks here!")


# User Input
input_query = st.text_input("Ask a question about Alenso:", placeholder="e.g., What services does Alenso offer?")

if input_query:
    # Perform Retrieval
    retrieved_knowledge = retrieve(input_query)
    
    # Update the sidebar dynamically to display retrieved chunks and their scores
    with context_placeholder.container():
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f"**Relevance (RRF):** `{similarity:.4f}`\n\n*{chunk}*")
            st.markdown("---")

    # Construct the instruction prompt
    instruction_prompt = f"""Rules, follow all of them:
1. Spelling: his name is "Alenso" - A-L-E-N-S-O, full name "Alen Alex". Never write "Alsenso", "Alenzo", "Allen", "Alonso" or anything else. Re-read your answer before finishing and fix the spelling if it's wrong.
2. You are Alenso's assistant, not Alenso himself - refer to him as "he"/"him"/"Alenso", never say "I" or "me" when talking about his work or offering to help.
3. Keep the answer short: 2-4 sentences, plain text, no bullet lists.
4. Fact-check yourself before answering: scan the context line by line for anything related to the question. If the context states a fact (even briefly, e.g. one item in a list), your answer must agree with it - never say "no" or "not" about something the context confirms is true. Do not paraphrase a fact into its opposite.
5. Use only the context below, don't invent facts that aren't there. Only if the topic is completely absent from the context, say briefly that you don't have that detail and tell the user to book a call with Alenso at alensocreations@gmail.com or via his website alenso.icu to get the answer.
6. If they are asking for social media handles - it's "alenso0"

Context:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
"""

    st.subheader("Response:")
    
    # Dynamic placeholder for streaming the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )
        
        # Iterate over stream chunks and dynamically update the UI
        for chunk in stream:
            token = chunk['message']['content']
            full_response += token
            response_placeholder.markdown(full_response + "▌")
            
        # Final update to remove the cursor icon
        response_placeholder.markdown(full_response)
        
    except Exception as e:
        st.error(f"Error generating chat response: {e}")