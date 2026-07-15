import streamlit as st
import ollama
import os
import re

st.set_page_config(page_title="Ollama RAG Demo",layout="centered")
st.title("Alenso's myBOT Demo")
st.write("This is a simple demo of how Alenso will take a user query, retrieve relevant chunks of information from a vector database, and then use those chunks as context to generate a response from a language model.")

# Model Configurations
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

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
    # embedding covers a complete thought instead of a fragment of one.
    dataset = re.split(r'\n\s*\n', raw_text)

    vector_db = []

    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i, chunk in enumerate(dataset):
        # Collapse the hard-wrapped lines within a paragraph into one clean string.
        chunk = ' '.join(line.strip() for line in chunk.splitlines() if line.strip())
        if not chunk:
            continue
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

# Initialize the database
with st.spinner("Initializing Vector Database and generating embeddings..."):
    VECTOR_DB = initialize_vector_db()


# --- 2. Helper Functions ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if int(norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)

## Retrieval function that takes a user query, computes its embedding, and finds the most similar chunks in the VECTOR_DB based on cosine similarity.
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


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
            st.markdown(f"**Score:** `{similarity:.2f}`\n\n*{chunk}*")
            st.markdown("---")

    # Construct the instruction prompt
    instruction_prompt = f"""You are Alenso's personal assistant chatbot, talking directly to a visitor who wants to know if Alenso is the right fit for them. Speak to the user as "you", not in the third person about "your business" in the abstract - make it feel like a conversation, not a brochure.

Naming rule: his name is always "Alen Alex", also known as "Alenso". Spell both exactly this way every time - never alter, misspell, or vary them (not "Alenzo", "Allen", "Alonso", etc).

Use only the context below to answer, and don't invent information that isn't there. If the context doesn't answer the user's question, don't just say you don't know - instead, tell the user you don't have that specific detail and invite them to book a call/meeting with Alenso directly to discuss it, sharing his contact email (alensocreations@gmail.com) or phone (+91-9995229833).

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