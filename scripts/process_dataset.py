import sys
import json
import os
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from supabase import create_client
from PyPDF2 import PdfReader
from dataclasses import dataclass, field
import openpyxl
import xlrd  # Ensure xlrd is installed for .xls files
import openai
from openai import OpenAI
import requests
import numpy as np

# Load environment variables for local testing
if os.getenv("GITHUB_ACTIONS") is None:  # Detect if running locally
    from dotenv import load_dotenv
    load_dotenv()

# Initialize Supabase and OpenAI
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SERVICE_ROLE")
supabase = create_client(supabase_url, supabase_key)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" 
client = OpenAI()
MAX_TOKENS = 8191




@dataclass
class ProcessedData:
    rows: List[Dict[str, Any]]
    aggregated_embedding: List[float]
    schema: Dict[str, Any]
    tags: List[Dict[str, str]]

@dataclass
class ProcessedChunks:
    rows: List[Dict[str, Any]]
    embeddings: List[float]


class FileHandler(ABC):
    def __init__(self, file_path: str, dataset_id: str, chunk_size: int, batch_size: int, tpm_limit: int):
        self.file_path = file_path
        self.dataset_id = dataset_id
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.tpm_limit = tpm_limit
        self.schema = {}
        self.tags = []
        self.all_rows = []
        self.all_embeddings = []

    @abstractmethod
    def process(self) -> ProcessedData:
        pass

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> ProcessedChunks :
        data = generate_embeddings_with_rate_limit(
            chunks=chunks,
            model=OPENAI_EMBEDDING_MODEL,
            tpm_limit=self.tpm_limit
        )
        return data


class CSVHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            # Read the entire CSV file into a single DataFrame
            dataframe = pd.read_csv(self.file_path)
            print(f"Processing entire CSV file")

            # Extract schema and tags if not already set
            if not self.schema and not self.tags:
                self.schema = {"fields": [{"name": col, "type": str(dtype)} for col, dtype in dataframe.dtypes.items()]}
                self.tags = [{"name": col} for col in dataframe.columns]

            # Create content rows and generate embeddings
            rows = create_content_rows(dataframe, self.dataset_id)
            chunks  = smart_chunk_data(rows, 1000)
            processedData = self.generate_embeddings(chunks)
            self.all_rows.extend(processedData.rows)
            self.all_embeddings.extend(processedData.embeddings)
            # Aggregate embeddings
            aggregated_embedding = aggregate_embeddings(self.all_embeddings)
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregated_embedding,
                schema=self.schema,
                tags=self.tags
            )
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            raise


class ExcelHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            # Determine the appropriate engine based on file extension
            file_ext = Path(self.file_path).suffix.lower()
            engine = 'xlrd' if file_ext == '.xls' else 'openpyxl'
            
            # Read the entire Excel file into a single DataFrame
            dataframe = pd.read_excel(self.file_path, engine=engine)
            print(f"Processing entire Excel file")

            # Extract schema and tags if not already set
            if not self.schema and not self.tags:
                self.schema = {"fields": [{"name": col, "type": str(dtype)} for col, dtype in dataframe.dtypes.items()]}
                self.tags = [{"name": col} for col in dataframe.columns]

            # Create content rows and generate embeddings
            #clean data 
            rows = create_content_rows(dataframe, self.dataset_id)
            chunks  = smart_chunk_data(rows, 1000)
            processedData = self.generate_embeddings(chunks)
            self.all_rows.extend(processedData.rows)
            self.all_embeddings.extend(processedData.embeddings)

            # Aggregate embeddings
            aggregated_embedding = aggregate_embeddings(self.all_embeddings)
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregated_embedding,
                schema=self.schema,
                tags=self.tags
            )
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            raise

class PDFHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            reader = PdfReader(self.file_path)
            content = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            chunks_text = split_text(content, 256)
            chunks = [{
                "dataset_id": self.dataset_id,
                "content": chunk,
                "metadata": {}
            } for chunk in chunks_text]
            data = self.generate_embeddings(chunks)
            self.all_rows.extend(data.rows)
            self.all_embeddings.extend(data.embeddings)
            
            aggregated_embedding = aggregate_embeddings(self.all_embeddings)
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregated_embedding,
                schema={},
                tags=[]
            )
        except Exception as e:
            print(f"Error processing PDF file: {e}")
            raise


class TextHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
            chunks_text = split_text(content, 256)
            chunks = [{
                "dataset_id": self.dataset_id,
                "content": chunk,
                "metadata": {}
            } for chunk in chunks_text]
            
            
            data = self.generate_embeddings(chunks)
            self.all_rows.extend(data.rows)
            self.all_embeddings.extend(data.embeddings)
            
            aggregated_embedding = aggregate_embeddings(self.all_embeddings)
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregated_embedding,
                schema={},
                tags=[]
            )
        except Exception as e:
            print(f"Error processing Text/Markdown file: {e}")
            raise


class DatasetProcessor:
    def __init__(self, payload: Dict[str, Any]):
        self.dataset_id = payload["id"]
        self.uri = payload["URI"]
        self.chunk_size = 1000
        self.batch_size = 50
        self.tpm_limit = 1000000
        self.file_path = download_file1(self.uri)
        self.file_ext = Path(self.file_path).suffix.lower()
        self.handler = self.get_handler()

    def get_handler(self) -> FileHandler:
        chunk_size, batch_size = dynamic_chunk_batch_sizes(self.file_path)
        if self.file_ext == ".csv":
            return CSVHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=chunk_size,
                batch_size=batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext in [".xls", ".xlsx"]:
            file_type = 'xls' if self.file_ext == '.xls' else 'xlsx'
            return ExcelHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=chunk_size,
                batch_size=batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext == ".pdf":
            return PDFHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=chunk_size,
                batch_size=batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext in [".md", ".txt"]:
            return TextHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=chunk_size,
                batch_size=batch_size,
                tpm_limit=self.tpm_limit
            )
        else:
            raise ValueError(f"Unsupported file extension: {self.file_ext}")

    def process(self) -> ProcessedData:
        print(f"Processing dataset {self.dataset_id} with file {self.file_path}")
        return self.handler.process()


# Helper Functions

# Function to download file
def download_file1(uri, destination="downloads"):
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


 # Function for dyamnic chunk sizes
def dynamic_chunk_batch_sizes(file_path: str):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 100:
        return 300, 50
    elif file_size_mb > 50:
        return 100, 25
    elif file_size_mb > 25:
        return 50, 10
    elif file_size_mb > 20:
        return 50, 10
    else:
        return 10, 5
    


def smart_chunk_data(data: List[Dict[str, Any]], max_tokens: int, max_chunks: int = 1500) -> List[Dict[str, Any]]:
    """
    Chunk data into a flat list of aggregated chunks, not exceeding max_chunks.
    Each chunk's total tokens do not exceed max_tokens.

    Args:
        data (List[Dict[str, Any]]): The list of data rows.
        max_tokens (int): Maximum number of tokens per chunk.
        max_chunks (int): Maximum number of chunks.

    Returns:
        List[Dict[str, Any]]: A list of aggregated data chunks.
    """
    total_tokens = sum(len(row.get("content", "").split()) for row in data)
    tokens_per_chunk = max(total_tokens // max_chunks, max_tokens)
    
    chunks = []
    current_chunk = {"dataset_id":"", "content": "", "metadata": {}}
    current_tokens = 0

    for row in data:
        content = row.get("content", "").strip()
        id = row.get("dataset_id", "")
        token_count = len(content.split())

        if current_tokens + token_count > tokens_per_chunk and len(chunks) < max_chunks - 1:
            chunks.append(current_chunk)
            current_chunk = {"dataset_id":"","content": "", "metadata": {}}
            current_tokens = 0
        current_chunk["dataset_id"] = id
        current_chunk["content"] += f" {content}"
        current_tokens += token_count
        current_chunk["metadata"] = {
            "word_count": len(current_chunk["content"].split()),
            "char_count": len(current_chunk["content"]),
            # Add more metadata fields as needed
        }

    return chunks

def generate_embeddings_with_rate_limit(chunks:List[Dict[str, Any]], model: str, tpm_limit: int) ->ProcessedChunks:
    total_tokens = 0
    start_time = time.time()
    embeddings = []
    count=0
    print(f"Number of chunks" + str(len(chunks)))
    for chunk in chunks:
        content = chunk["content"] if chunk["content"].strip() else " "
        token_count = len(content.split())
        total_tokens += token_count

        # Check if rate limit exceeded
        elapsed = time.time() - start_time
        if elapsed < 60 and total_tokens > tpm_limit:
            wait_time = 60 - elapsed
            print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            start_time = time.time()
            total_tokens = 0

        try:
            response = client.embeddings.create(input=[content], model=model)
            embedding = response.data[0].embedding
            if "embedding" not in chunk:
                    chunk["embedding"] = []  # Initialize if not present
            chunk["embedding"] = embedding
            embeddings.append(embedding)

            print(f"Generated embedding for chunk" + str(count))
            count+=1
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            raise

    return ProcessedChunks(rows=chunks, embeddings=embeddings)


# Function to aggregate embeddings
def aggregate_embeddings(embeddings):
    """
    Compute an aggregated embedding for the dataset by averaging.
    Ensure the aggregated embedding does not exceed 1536 dimensions.
    """
    if not embeddings:
        return [0] * 1536  # Return a zero vector if no embeddings

    np_embeddings = np.array(embeddings)

    if (np_embeddings.shape[0]>1)  and np_embeddings.shape[0] > 1536:
        # Ensure embeddings are sliced properly for high dimensions
        np_embeddings = np_embeddings[:1536]

    # If the array is already one-dimensional, return as-is
    if len(np_embeddings.shape) == 1:
        return np_embeddings.tolist()

    return np.mean(np_embeddings, axis=0).tolist()
    
  

def chunk_data(data: List[Dict[str, Any]], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def create_content_rows(dataframe: pd.DataFrame, dataset_id: str) -> List[Dict[str, Any]]:
    def sanitise_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NaN and NaT values to None for JSON compatibility."""
        return {
        k: (v.isoformat() if isinstance(v, pd.Timestamp) else None if pd.isna(v) else v)
        for k, v in metadata.items()
    }
    rows = []
    for _, row in dataframe.iterrows():
        content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
        sanitised_metadata = sanitise_metadata(row.to_dict())
        rows.append({
            "dataset_id": dataset_id,
            "content": content,
            "metadata": sanitised_metadata
        })
    return rows

def attach_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]  # Attach the embedding directly to the chunk
    return chunks  # Return the updated chunks as rows


def update_supabase_dataset(dataset_id: str, schema: Dict[str, Any], tags: List[Dict[str, str]], embedding: List[float]):
    try:
        response = supabase.table("datasets").update({
            "schema": json.dumps(schema),
            "tags": json.dumps(tags),
            "embedding": embedding
        }).eq("id", dataset_id).execute()

        if not response.data:
            raise Exception(f"Error updating dataset: {response}")
        print("Supabase dataset update successful!")
    except Exception as e:
        print(f"Error updating Supabase dataset: {e}")
        raise

def split_into_batches(data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Splits a list of data into smaller batches.

    Args:
        data (List[Dict[str, Any]]): The list of data rows to split.
        batch_size (int): The number of rows per batch.

    Returns:
        List[List[Dict[str, Any]]]: A list containing batched data rows.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def insert_rows_into_supabase(rows: List[Dict[str, Any]],batch_size: int = 100,max_retries: int = 3,backoff_factor: float = 0.5):
    batches = split_into_batches(rows, batch_size)
    total_batches = len(batches)
    print(f"Total batches to upsert: {total_batches}")

    for idx, batch in enumerate(batches, start=1):
        attempt = 0
        while attempt <= max_retries:
            try:
                response = supabase.table("dataset_rows").upsert(batch).execute()
                print(f"Batch {idx}/{total_batches} upserted successfully! Count: {response.count}")
                break  # Exit retry loop on success
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"Failed to insert batch {idx}/{total_batches} after {max_retries} attempts: {e}")
                    raise
                wait_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Error inserting batch {idx}/{total_batches}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)


def split_text(content: str, chunk_size: int) -> List[str]:
    """
    Splits the given text into chunks of specified size.

    Args:
        content (str): The text to split.
        chunk_size (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list containing text chunks.
    """
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Main Processing Function

def process_dataset(payload: Dict[str, Any]):
    try:
        
        processor = DatasetProcessor(payload)
        processed_data = processor.process()
        if len(processed_data.rows)>0:
            
            # Update dataset metadata in Supabase
            update_supabase_dataset(
                dataset_id=processor.dataset_id,
                schema=processed_data.schema,
                tags=processed_data.tags,
                embedding=processed_data.aggregated_embedding
            )

            # Insert rows into Supabase
            insert_rows_into_supabase(processed_data.rows)

        print(f"Successfully processed dataset {processor.dataset_id}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        sys.exit(1)


# Entry Point

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_dataset.py <payload.json>")
        sys.exit(1)
    payload_file = sys.argv[1]
    try:
        with open(payload_file, 'r') as f:
            print("Loading payload file...")
            print(f"Payload file: {payload_file}")
            payload = json.load(f)
        process_dataset(payload)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Payload file not found: {payload_file}")
        sys.exit(1)