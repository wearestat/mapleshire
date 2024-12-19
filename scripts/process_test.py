mport sys
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

