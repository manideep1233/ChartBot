import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
FAISS_DIR = "faiss_db"

def build_vector_database():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}' folder. Drop your exam books/PDFs inside it.")
        return

    files = os.listdir(DATA_DIR)
    if not files:
        print(f"⚠️ The '{DATA_DIR}' folder is empty. Place your study materials there first!")
        return

    print(f"📚 Found {len(files)} file(s) in '{DATA_DIR}'. Loading pages...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} raw pages.")

    print("✂️ Splitting text into manageable chunks while preserving page numbers...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(raw_documents)
    total_chunks = len(chunks)
    print(f"🧩 Created {total_chunks} total text chunks.")

    print("🧠 Initializing nomic-embed-text embedding engine...")
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # SAFE BATCH PROCESSING LAYER
    # Process 50 chunks at a time so your local machine's RAM and ports don't overflow
    batch_size = 50
    vector_store = None
    
    print(f"🚀 Starting secure batch processing (Size: {batch_size} chunks per batch)...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"📦 Processing chunks {i} to {min(i + batch_size, total_chunks)} of {total_chunks}...")
        
        try:
            if vector_store is None:
                # Initialize the FAISS database with the very first batch
                vector_store = FAISS.from_documents(batch, embeddings_model)
            else:
                # Add subsequent batches to the existing index structure
                vector_store.add_documents(batch)
        except Exception as batch_error:
            print(f"❌ Batch processing failed at chunk index {i}. Error details: {batch_error}")
            print("💡 Tip: Try running the taskkill command and restarting Ollama if the connection broke.")
            return

    if vector_store:
        # Save the finalized compiled index cleanly onto your hard drive
        vector_store.save_local(FAISS_DIR)
        print(f"\n🎉 Success! Local FAISS store permanently saved to '{FAISS_DIR}'.")
    else:
        print("❌ Database building was interrupted; no files written.")

if __name__ == "__main__":
    build_vector_database()