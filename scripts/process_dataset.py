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

# Initialize OpenAI and Supabase
# Load environment variables from .env file
load_dotenv()
client = OpenAI()
supabase_url = os.getenv("PRIVATE_SUPABASE_URL")
supabase_key = os.getenv("PRIVATE_SUPABASE_ANON_KEY")
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

# Process CSV files
def process_csv(file_path):
    dataframe = pd.read_csv(file_path)
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    tags = dataframe.columns.tolist()
    content = dataframe.head(100).to_markdown(index=False)  # Convert first 100 rows to Markdown
    return content, schema, tags

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
    response = supabase.table("datasets").update({
        "schema": schema,
        "tags": tags,
        "embedding": embedding
    }).eq("id", dataset_id).execute()
    if response.error:
        raise Exception(f"Supabase error: {response.error}")

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
