import sys
import json
import os
import time
from pathlib import Path
import PyPDF2
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from supabase import create_client
from dataclasses import dataclass, field
import openai
from openai import OpenAI
import requests
import numpy as np
from llama_index import Document
from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, Document
from langchain.chat_models import ChatOpenAI

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



#Classes 
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
        self.schema = {}
        self.tags = []
        self.all_rows = []
        self.all_embeddings = []

    @abstractmethod
    def process(self) -> ProcessedData:
        pass

    def generate_embeddings(self) -> ProcessedChunks :
        data =""
        return data


class CSVHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregate_embeddings(self.all_embeddings),
                schema=self.schema,
                tags=self.tags
            )
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            raise


class ExcelHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregate_embeddings(self.all_embeddings),
                schema=self.schema,
                tags=self.tags
            )
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            raise


class PDFHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            reader = pd.read_csv(self.file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            document = Document(text=content)
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregate_embeddings(self.all_embeddings),
                schema={},
                tags=[]
            )
        except Exception as e:
            print(f"Error processing PDF file: {e}")
            raise


class TextHandler(FileHandler):
    def process(self) -> ProcessedData:
        try:
            
            return ProcessedData(
                rows=self.all_rows,
                aggregated_embedding=aggregate_embeddings(self.all_embeddings),
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
        self.file_path = download_file1(self.uri)
        self.file_ext = Path(self.file_path).suffix.lower()
        self.handler = self.get_handler()

    def get_handler(self) -> FileHandler:
        if self.file_ext == ".csv":
            return CSVHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
              
            )
        elif self.file_ext in [".xls", ".xlsx"]:
            file_type = 'xls' if self.file_ext == '.xls' else 'xlsx'
            return ExcelHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,

            )
        elif self.file_ext == ".pdf":
            return PDFHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                
            )
        elif self.file_ext in [".md", ".txt"]:
            return TextHandler(
                file_path=self.file_path,
                dataset_id=self.dataset_id,
                
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
    
# Function to update supabase dataset
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

# Function to insert dataset rows into supabase
def insert_rows_into_supabase(rows: List[Dict[str, Any]]):
    try:
        response = supabase.table("dataset_rows").upsert(rows).execute()
        print(f"Rows successfully inserted into dataset_rows! Count: {response.count}")
    except Exception as e:
        print(f"Error inserting rows into dataset_rows: {e}")
        raise


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


