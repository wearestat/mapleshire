import os
import json
import sys
import requests
import pandas as pd
from pathlib import Path
from supabase import create_client
from PyPDF2 import PdfReader
import numpy as np
from openai import OpenAI

# Load environment variables for local testing
if os.getenv("GITHUB_ACTIONS") is None:  # Detect if running locally
    from dotenv import load_dotenv
    load_dotenv()

# Initialize OpenAI and Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SERVICE_ROLE")
supabase = create_client(supabase_url, supabase_key)
MAX_TOKENS = 8191
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()

# Function to download file
def download_file(uri, destination="downloads"):
    os.makedirs(destination, exist_ok=True)
    file_name = Path(uri).name
    file_path = os.path.join(destination, file_name)

    # Convert GitHub blob URL to raw URL if needed
    if "github.com" in uri:
        uri = uri.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(uri)
    response.raise_for_status()
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path

# Generate embedding for a single input
def generate_embedding(content):
    response = client.embeddings.create(
        input=content,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Aggregate embeddings by averaging
def aggregate_embeddings(embeddings):
    return np.mean(embeddings, axis=0).tolist()

# Generate embeddings for chunks
def generate_embeddings_for_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        if len(chunk) > MAX_TOKENS:
            chunk = chunk[:MAX_TOKENS]  # Truncate to avoid exceeding token limit
        try:
            response = client.embeddings.create(input=chunk, model=OPENAI_EMBEDDING_MODEL)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            raise
    return embeddings

def process_csv_with_batching(file_path, dataset_id, chunk_size=100, batch_size=1000):
    """
    Process a large CSV file with batching and chunking.
    Each chunk is a group of rows combined into one text block.
    """
    dataframe = pd.read_csv(file_path)
    
    # Extract schema
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    tags = [{"name": col} for col in dataframe.columns]
    
    chunks = []  # List to store chunks
    embeddings = []  # List to store all embeddings

    # Step 1: Create chunks of rows
    for i in range(0, len(dataframe), chunk_size):
        chunk = dataframe.iloc[i:i + chunk_size]
        chunk_content = "\n".join([
            " ".join([f"{col}: {row[col]}" for col in chunk.columns if pd.notna(row[col])])
            for _, row in chunk.iterrows()
        ])
        chunks.append({
            "dataset_id": dataset_id,
            "content": chunk_content,
            "metadata": {"chunk_start": i, "chunk_end": min(i + chunk_size, len(dataframe))}
        })
    
    # Step 2: Batch embeddings
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            # Prepare input for embedding API
            batch_contents = [chunk["content"] for chunk in batch]
            response = client.embeddings.create(input=batch_contents, model=OPENAI_EMBEDDING_MODEL)
            batch_embeddings = [data["embedding"] for data in response.data]

            # Attach embeddings to chunks
            for j, embedding in enumerate(batch_embeddings):
                batch[j]["embedding"] = embedding
                embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
            raise

    # Step 3: Compute aggregated embedding for the entire dataset
    aggregated_embedding = aggregate_embeddings(embeddings)

    return chunks, aggregated_embedding, schema, tags

# Process CSV files
def process_csv(file_path, dataset_id, chunk_size=1000):
    dataframe = pd.read_csv(file_path)
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    tags = [{"name": col} for col in dataframe.columns]
    
    rows = []
    embeddings = []
    
    for index, row in dataframe.iterrows():
        content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
        embedding = generate_embedding(content)
        rows.append({
            "dataset_id": dataset_id,
            "content": content,
            "embedding": embedding,
            "metadata": row.to_dict()
        })
        embeddings.append(embedding)

    # Calculate aggregated embedding
    aggregated_embedding = aggregate_embeddings(embeddings)
    return rows, aggregated_embedding, schema, tags

# Process PDF files
def process_pdf(file_path, dataset_id, chunk_size=1000):
    reader = PdfReader(file_path)
    content = " ".join(page.extract_text() for page in reader.pages)
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    embeddings = generate_embeddings_for_chunks(chunks)

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "dataset_id": dataset_id,
            "content": chunk,
            "embedding": embeddings[i],
            "metadata": {}
        })

    aggregated_embedding = aggregate_embeddings(embeddings)
    schema = None
    tags = []
    return rows, aggregated_embedding, schema, tags

# Process text or Markdown files
def process_text_or_markdown(file_path, dataset_id, chunk_size=1000):
    with open(file_path, "r") as file:
        content = file.read()
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    embeddings = generate_embeddings_for_chunks(chunks)

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "dataset_id": dataset_id,
            "content": chunk,
            "embedding": embeddings[i],
            "metadata": {}
        })

    aggregated_embedding = aggregate_embeddings(embeddings)
    schema = None
    tags = []
    return rows, aggregated_embedding, schema, tags

# Update dataset metadata in Supabase
def update_supabase_dataset(dataset_id, schema, tags, embedding):
    response = supabase.table("datasets").update({
        "schema": json.dumps(schema),
        "tags": json.dumps(tags),
        "embedding": embedding
    }).eq("id", dataset_id).execute()

    if not response.data:
        raise Exception(f"Error updating dataset: {response}")
    print("Supabase dataset update successful!")

# Insert rows into `dataset_rows` table in Supabase
def insert_rows_into_supabase(rows):
    for row in rows:
        response = supabase.table("dataset_rows").insert({
            "dataset_id": row["dataset_id"],
            "content": row["content"],
            "embedding": row["embedding"],
            "metadata": json.dumps(row["metadata"])
        }).execute()

        if response.error:
            raise Exception(f"Error inserting row: {response.error}")
    print("Rows successfully inserted into dataset_rows!")

# Main function to process datasets
def process_dataset(payload):
    try:
        # Parse payload
        print("Parsing JSON payload")
        with open(payload, 'r') as f:
            payload = json.load(f)
        dataset_id = payload["id"]
        uri = payload["URI"]

        print(f"Processing dataset {dataset_id}")
        file_path = download_file(uri)
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".csv": 
            # Process CSV with batching and chunking
            rows, aggregated_embedding, schema, tags = process_csv_with_batching(
                file_path=file_path,
                dataset_id=dataset_id,
                chunk_size=100,   # Customize the chunk size for row grouping
                batch_size=300   # Customize the batch size for embedding requests
            )
        elif file_ext == ".pdf":
            rows, aggregated_embedding, schema, tags = process_pdf(file_path, dataset_id)
        elif file_ext in [".md", ".txt"]:
            rows, aggregated_embedding, schema, tags = process_text_or_markdown(file_path, dataset_id)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Update dataset in `datasets` table
        update_supabase_dataset(dataset_id, schema, tags, aggregated_embedding)

        # Insert rows into `dataset_rows` table
        insert_rows_into_supabase(rows)

        print(f"Successfully processed dataset {dataset_id}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_dataset.py '<payload_file>'")
        sys.exit(1)

    payload_file = sys.argv[1]
    process_dataset(payload_file)
