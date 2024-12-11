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
import time
import openpyxl
import logging

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

# Function to chunk data
def chunk_data(data, chunk_size):
    """
    Split data into chunks of specified size.
    
    Args:
        data (list): List of data items.
        chunk_size (int): Number of items per chunk.
    
    Returns:
        generator: Yields chunks of data.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Function to create content rows
def create_content_rows(dataframe, dataset_id):
    """
    Create content rows from a DataFrame.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to process.
        dataset_id (str): Identifier for the dataset.
    
    Returns:
        list: List of dictionaries with content and metadata.
    """
    rows = []
    for _, row in dataframe.iterrows():
        content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
        rows.append({
            "dataset_id": dataset_id,
            "content": content,
            "metadata": row.to_dict()
        })
    return rows

# Function to generate embeddings for chunks
def generate_rows_with_embeddings(chunks, generate_embeddings_func, batch_size, model, tpm_limit):
    """
    Generate embeddings and attach them to content chunks.
    
    Args:
        chunks (list): List of content dictionaries.
        generate_embeddings_func (function): Function to generate embeddings.
        batch_size (int): Number of chunks per batch.
        model (str): OpenAI embedding model.
        tpm_limit (int): Tokens per minute limit.
    
    Returns:
        list: List of rows with embeddings.
    """
    embeddings = generate_embeddings_func(
        chunks=chunks,
        batch_size=batch_size,
        model=model,
        tpm_limit=tpm_limit
    )
    return [
        {
            "dataset_id": chunk["dataset_id"],
            "content": chunk["content"],
            "embedding": embeddings[i],
            "metadata": chunk["metadata"]
        }
        for i, chunk in enumerate(chunks)
    ]



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


def generate_embeddings_with_rate_limit(chunks, batch_size, model, tpm_limit):
    """
    Generate embeddings with rate limiting to respect OpenAI TPM constraints.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            # Prepare input for embedding API
            batch_contents = [chunk["content"] for chunk in batch]
            token_count = sum(len(content.split()) for content in batch_contents)

            # Ensure we don’t exceed TPM
            if token_count > tpm_limit:
                wait_time = token_count / tpm_limit * 60  # Calculate wait time in seconds
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            response = client.embeddings.create(input=batch_contents, model=model)
            batch_embeddings = [data.embedding for data in response.data]

            # Attach embeddings to chunks
            for j, embedding in enumerate(batch_embeddings):
                batch[j]["embedding"] = embedding
                embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
            raise

    return embeddings

def process_file(file_path, dataset_id, chunk_size=1000, batch_size=50, tpm_limit=1000000, file_type='csv'):
    """
    Generalized function to process CSV and XLS/XLSX files.
    
    Args:
        file_path (str): Path to the file.
        dataset_id (str): Identifier for the dataset.
        chunk_size (int): Number of rows per chunk.
        batch_size (int): Number of rows per batch for embedding generation.
        tpm_limit (int): Tokens per minute limit.
        file_type (str): Type of the file ('csv', 'xls', 'xlsx').
    
    Returns:
        dict: Contains rows, aggregated_embedding, schema, and tags.
    """
    try:
        if file_type == 'csv':
            reader = pd.read_csv(file_path, chunksize=chunk_size)
        else:
            engine = 'xlrd' if file_type == 'xls' else 'openpyxl'
            dataframe = pd.read_excel(file_path, engine=engine)
            reader = chunk_data(dataframe.to_dict('records'), chunk_size)
        
        all_rows = []
        all_embeddings = []
        schema = {}
        tags = []
        
        for chunk_number, data_chunk in enumerate(reader, 1):
            print(f"Processing chunk {chunk_number}")
            
            if file_type == 'csv':
                dataframe_chunk = data_chunk
                if not schema and not tags:
                    schema = {"fields": [{"name": col, "type": str(dataframe_chunk[col].dtype)} for col in dataframe_chunk.columns]}
                    tags = [{"name": col} for col in dataframe_chunk.columns]
                chunks = create_content_rows(dataframe_chunk, dataset_id)
            else:
                dataframe_chunk = pd.DataFrame(data_chunk)
                if not schema and not tags:
                    schema = {"fields": [{"name": col, "type": str(dataframe_chunk[col].dtype)} for col in dataframe_chunk.columns]}
                    tags = [{"name": col} for col in dataframe_chunk.columns]
                chunks = create_content_rows(dataframe_chunk, dataset_id)
            
            # Generate rows with embeddings
            rows = generate_rows_with_embeddings(
                chunks=chunks,
                generate_embeddings_func=generate_embeddings_with_rate_limit,
                batch_size=batch_size,
                model=OPENAI_EMBEDDING_MODEL,
                tpm_limit=tpm_limit
            )
            
            all_rows.extend(rows)
            all_embeddings.extend([row["embedding"] for row in rows])
        
        aggregated_embedding = aggregate_embeddings(all_embeddings)
        
        return {
            "rows": all_rows,
            "aggregated_embedding": aggregated_embedding,
            "schema": schema,
            "tags": tags
        }
    
    except Exception as e:
        print(f"Error processing file: {e}")
        raise

# Process text and PDF files with batching and chunking
def process_general_file(content, dataset_id, chunk_size=1000, batch_size=50, tpm_limit=1000000):
    """
    Generalized function to process PDF and Text/Markdown files.
    
    Args:
        content (str): Extracted text content.
        dataset_id (str): Identifier for the dataset.
        chunk_size (int): Number of characters per chunk.
        batch_size (int): Number of chunks per batch for embedding generation.
        tpm_limit (int): Tokens per minute limit.
    
    Returns:
        dict: Contains rows, aggregated_embedding, schema, and tags.
    """
    try:
        chunks_text = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        chunks = [{
            "dataset_id": dataset_id,
            "content": chunk,
            "metadata": {}
        } for chunk in chunks_text]
        
        rows = generate_rows_with_embeddings(
            chunks=chunks,
            generate_embeddings_func=generate_embeddings_with_rate_limit,
            batch_size=batch_size,
            model=OPENAI_EMBEDDING_MODEL,
            tpm_limit=tpm_limit
        )
        
        all_embeddings = [row["embedding"] for row in rows]
        aggregated_embedding = aggregate_embeddings(all_embeddings)
        
        return {
            "rows": rows,
            "aggregated_embedding": aggregated_embedding,
            "schema": None,
            "tags": []
        }
    
    except Exception as e:
        print(f"Error processing general file: {e}")
        raise

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

    response = supabase.table("dataset_rows").upsert(rows).execute()
    print("Rows successfully inserted into dataset_rows!" + response.count)

def process_dataset(payload):
    try:
        # Parse payload
        print("Parsing JSON payload")
        dataset_id = payload["id"]
        uri = payload["URI"]

        print(f"Processing dataset {dataset_id}")
        file_path = download_file(uri)
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".csv":
            processed_data = process_file(
                file_path=file_path,
                dataset_id=dataset_id,
                chunk_size=1000,
                batch_size=50,
                tpm_limit=1000000,
                file_type='csv'
            )
        elif file_ext in [".xls", ".xlsx"]:
            file_type = 'xls' if file_ext == '.xls' else 'xlsx'
            processed_data = process_file(
                file_path=file_path,
                dataset_id=dataset_id,
                chunk_size=1000,
                batch_size=50,
                tpm_limit=1000000,
                file_type=file_type
            )
        elif file_ext == ".pdf":
            reader = PdfReader(file_path)
            content = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            processed_data = process_general_file(
                content=content,
                dataset_id=dataset_id,
                chunk_size=1000,
                batch_size=50,
                tpm_limit=1000000
            )
        elif file_ext in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            processed_data = process_general_file(
                content=content,
                dataset_id=dataset_id,
                chunk_size=1000,
                batch_size=50,
                tpm_limit=1000000
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")

        # Update dataset metadata in Supabase
        update_supabase_dataset(
            dataset_id=dataset_id,
            schema=processed_data["schema"],
            tags=processed_data["tags"],
            embedding=processed_data["aggregated_embedding"]
        )

        # Insert rows into Supabase
        insert_rows_into_supabase(processed_data["rows"])

        print(f"Successfully processed dataset {dataset_id}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        sys.exit(1)

# Entry point 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_dataset.py <payload.json>")
        sys.exit(1)
    payload_file = sys.argv[1]
    try:
        with open(payload_file, 'r') as f:
            payload = json.load(f)
        process_dataset(payload)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Payload file not found: {payload_file}")
        sys.exit(1)