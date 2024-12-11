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

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        embeddings = generate_embeddings_with_rate_limit(
            chunks=chunks,
            batch_size=self.batch_size,
            model=OPENAI_EMBEDDING_MODEL,
            tpm_limit=self.tpm_limit
        )
        return embeddings


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
                embeddings = self.generate_embeddings(chunks)
                rows = attach_embeddings(chunks, embeddings)
                self.all_rows.extend(rows)
                self.all_embeddings.extend(embeddings)
            
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
                    self.schema = {"fields": [{"name": col, "type": str(dtype)} for col, dtype in dataframe_chunk.dtypes.iteritems()]}
                    self.tags = [{"name": col} for col in dataframe_chunk.columns]
                
                chunks = create_content_rows(dataframe_chunk, self.dataset_id)
                embeddings = self.generate_embeddings(chunks)
                rows = attach_embeddings(chunks, embeddings)
                self.all_rows.extend(rows)
                self.all_embeddings.extend(embeddings)
            
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
            
            embeddings = self.generate_embeddings(chunks)
            rows = attach_embeddings(chunks, embeddings)
            self.all_rows.extend(rows)
            self.all_embeddings.extend(embeddings)
            
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
            
            embeddings = self.generate_embeddings(chunks)
            rows = attach_embeddings(chunks, embeddings)
            self.all_rows.extend(rows)
            self.all_embeddings.extend(embeddings)
            
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
        self.file_path = download_file(self.uri)
        self.file_ext = Path(self.file_path).suffix.lower()
        self.handler = self.get_handler()

    def get_handler(self) -> FileHandler:
        if self.file_ext == ".csv":
            return CSVHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext in [".xls", ".xlsx"]:
            file_type = 'xls' if self.file_ext == '.xls' else 'xlsx'
            return ExcelHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext == ".pdf":
            return PDFHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                tpm_limit=self.tpm_limit
            )
        elif self.file_ext in [".md", ".txt"]:
            return TextHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                tpm_limit=self.tpm_limit
            )
        else:
            raise ValueError(f"Unsupported file extension: {self.file_ext}")

    def process(self) -> ProcessedData:
        print(f"Processing dataset {self.dataset_id} with file {self.file_path}")
        return self.handler.process()


# Helper Functions

def download_file(uri: str, destination: str = "downloads") -> str:
    import requests
    from urllib.parse import urlparse

    os.makedirs(destination, exist_ok=True)
    parsed_url = urlparse(uri)
    filename = os.path.basename(parsed_url.path)
    file_path = os.path.join(destination, filename)

    try:
        response = requests.get(uri, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise

def generate_embeddings_with_rate_limit(chunks: List[Dict[str, Any]], batch_size: int, model: str, tpm_limit: int) -> List[List[float]]:
    embeddings = []
    total_tokens = 0
    start_time = time.time()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_contents = [chunk["content"] for chunk in batch]
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
            print(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i + batch_size}: {e}")
            raise

    return embeddings

def aggregate_embeddings(embeddings: List[List[float]]) -> List[float]:
    import numpy as np
    if not embeddings:
        return []
    return list(np.mean(embeddings, axis=0))

def chunk_data(data: List[Dict[str, Any]], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def create_content_rows(dataframe: pd.DataFrame, dataset_id: str) -> List[Dict[str, Any]]:
    rows = []
    for _, row in dataframe.iterrows():
        content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
        rows.append({
            "dataset_id": dataset_id,
            "content": content,
            "metadata": row.to_dict()
        })
    return rows

def attach_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    return [
        {
            "dataset_id": chunk["dataset_id"],
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk["metadata"]
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]


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


def attach_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    return [
        {
            "dataset_id": chunk["dataset_id"],
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk["metadata"]
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

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
    if sys.stdin.isatty():
        print("Usage: python process_dataset.py < payload.json")
        sys.exit(1)
    else:
        payload_input = sys.stdin.read()
        print(f"Received payload: {payload_input}")  # Debugging line
        try:
            payload = json.loads(payload_input)
            process_dataset(payload)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON input: {e}")
            sys.exit(1)