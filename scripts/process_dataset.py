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
            batch_size=self.batch_size,
            model=OPENAI_EMBEDDING_MODEL,
            tpm_limit=self.tpm_limit
        )
        return data


class CSVHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            reader = pd.read_csv(self.file_path, chunksize=self.chunk_size)
            for chunk_number, dataframe in enumerate(reader, 1):
                print(f"Processing CSV chunk {chunk_number}")
                if not self.schema and not self.tags:
                    self.schema = {"fields": [{"name": col, "type": str(dtype)} for col, dtype in dataframe.dtypes.iteritems()]}
                    self.tags = [{"name": col} for col in dataframe.columns]
                
                chunks = create_content_rows(dataframe, self.dataset_id)
                data = self.generate_embeddings(chunks)
                self.all_rows.extend(data.rows)
                self.all_embeddings.extend(data.embeddings)
            
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
            file_ext = Path(self.file_path).suffix.lower()
            engine = 'xlrd' if file_ext == '.xls' else 'openpyxl'
            dataframe = pd.read_excel(self.file_path, engine=engine)
            for chunk_number, data_chunk in enumerate(chunk_data(dataframe.to_dict('records'), self.chunk_size), 1):
                print(f"Processing Excel chunk {chunk_number}")
                dataframe_chunk = pd.DataFrame(data_chunk)
                if not self.schema and not self.tags:
                    self.schema = {"fields": [{"name": col, "type": str(dtype)} for col, dtype in dataframe_chunk.dtypes.items()]}
                    self.tags = [{"name": col} for col in dataframe_chunk.columns]
                
                chunks = create_content_rows(dataframe_chunk, self.dataset_id)
                data = self.generate_embeddings(chunks)
                self.all_rows.extend(data.rows)
                self.all_embeddings.extend(data.embeddings)
            
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
            chunks_text = split_text(content, self.chunk_size)
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
            chunks_text = split_text(content, self.chunk_size)
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
        self.chunk_size = 500
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
        return 100, 25
    elif file_size_mb > 10:
        return 500, 50
    else:
        return 1000, 100
    

def generate_embeddings_with_rate_limit(chunks: List[Dict[str, Any]], batch_size: int, model: str, tpm_limit: int) ->ProcessedChunks:
    total_tokens = 0
    start_time = time.time()
    embeddings=[]
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_contents = [chunk["content"] if chunk["content"].strip() else " " for chunk in batch] 
        token_count = sum(len(content.split()) for content in batch_contents)
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
            response = client.embeddings.create(input=batch_contents, model=model)
            batch_embeddings = response.data[0].embedding
            embeddings.extend(batch_embeddings)
            # Attach embeddings directly to chunks
            for j, chunk in enumerate(chunks):
                if "embedding" not in chunk:
                    chunk["embedding"] = []  # Initialize if not present
                chunk["embedding"].append(batch_embeddings[j])
            print(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i + batch_size}: {e}")
            raise

    #return updated chunks which are the rows for the dataset
    return ProcessedChunks(
                rows=chunks,
                embeddings=embeddings
            )

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
        return {k: (None if pd.isna(v) else v) for k, v in metadata.items()}
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
    """
    Attach embeddings to chunks by averaging embeddings if multiple embeddings exist per chunk.
    """
    for i, chunk in enumerate(chunks):
        # Validate and attach embedding directly
        embedding = embeddings[i]
        chunk["embedding"] = embedding

    return chunks

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

def insert_rows_into_supabase(rows: List[Dict[str, Any]]):
    try:
        response = supabase.table("dataset_rows").upsert(rows).execute()
        print(f"Rows successfully inserted into dataset_rows! Count: {response.count}")
    except Exception as e:
        print(f"Error inserting rows into dataset_rows: {e}")
        raise



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
            payload = json.load(f)
        process_dataset(payload)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Payload file not found: {payload_file}")
        sys.exit(1)