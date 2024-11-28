from dotenv import load_dotenv
import os
import json
import sys
import requests
import pandas as pd
from pathlib import Path
from supabase import create_client
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np

# Initialize OpenAI and Supabase
# Load environment variables from .env file
load_dotenv()
client = OpenAI()
supabase_url = os.getenv("PRIVATE_SUPABASE_URL")
supabase_key = os.getenv("SERVICE_ROLE")
supabase = create_client(supabase_url, supabase_key)

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

def aggregate_embeddings(embeddings):
    """
    Aggregate embeddings by averaging them.
    :param embeddings: List of embeddings (each is a list of floats)
    :return: Averaged embedding
    """
    return np.mean(embeddings, axis=0).tolist()  # Average across all dimensions

def generate_embeddings_for_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = generate_embedding(chunk)  # Existing function to generate embedding
        embeddings.append(embedding)
    return embeddings


def process_csv(file_path, chunk_size=1000):
    dataframe = pd.read_csv(file_path)

    # Extract schema
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    
    # Extract tags (column names as tags)
    tags = [{"name": col} for col in dataframe.columns]

    # Generate chunks
    chunks = []
    for i in range(0, len(dataframe), chunk_size):
        chunk = dataframe.iloc[i:i + chunk_size]
        chunk_content = chunk.to_markdown(index=False)
        chunks.append(chunk_content)

    # Generate embeddings for each chunk
    embeddings = generate_embeddings_for_chunks(chunks)

    # Compute the averaged embedding
    aggregated_embedding = aggregate_embeddings(embeddings)

    return aggregated_embedding, schema, tags



# Process PDF files
def process_pdf(file_path):
    reader = PdfReader(file_path)
    content = " ".join(page.extract_text() for page in reader.pages)
    schema = None  # PDFs typically don't have a schema
    tags = []  # Tags could be extracted with NLP tools for topics
    return content, schema, tags

# Process Markdown/Text files
def process_text_or_markdown(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    schema = None  # Text and Markdown files typically don't have a schema
    tags = []  # Tags could be extracted with NLP tools
    return content, schema, tags

# Generate embeddings
def generate_embedding(content):

    response = client.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Update Supabase
def update_supabase(dataset_id, schema, tags, embedding):
    # Perform the update
    response = supabase.table("datasets").update({
        "schema": json.dumps(schema),  # Convert schema to JSON string
        "tags": json.dumps(tags),  # Convert tags to JSON string
        "embedding": embedding
    }).eq("id", dataset_id).execute()

    # Check if there was an error
    if not response.data:
        raise Exception(f"Error updating dataset: {response}")
    print("Supabase update successful!")

# Main function to process files
def process_dataset(payload):
    try:
        # Parse payload
        payload = json.loads(payload)
        dataset_id = payload["id"]
        organisation_id = payload["organisation_id"]
        uri = payload["URI"]

        print(f"Processing dataset {dataset_id} for organisation {organisation_id}")

        # Step 1: Download file
        file_path = download_file(uri)

        # Step 2: Determine file type and process
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".csv":
            content, schema, tags = process_csv(file_path)
        elif file_ext == ".pdf":
            content, schema, tags = process_pdf(file_path)
        elif file_ext in [".md", ".txt"]:
            content, schema, tags = process_text_or_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Step 3: Generate embedding
        embedding = generate_embedding(content)

        # Step 4: Update Supabase
        update_supabase(dataset_id, schema, tags, embedding)

        print(f"Successfully processed dataset {dataset_id}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    # Pass payload as an argument to the script
    if len(sys.argv) != 2:
        print("Usage: python process_dataset.py '<payload>'")
        sys.exit(1)
    process_dataset(sys.argv[1])
