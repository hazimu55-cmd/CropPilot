from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
import os
import re
import pandas as pd
from io import StringIO


def extract_tables_from_text(text: str) -> list:
    """
    Extract tables from text using heuristics
    Returns list of tuples (table_text, start_pos, end_pos)
    """
    tables = []
    lines = text.split('\n')
    
    # Look for table-like patterns (multiple lines with similar structure)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if line looks like a table row (has multiple columns separated by tabs or pipes)
        if '\t' in line or '|' in line:
            # Found potential table start
            table_start = i
            table_lines = [line]
            i += 1
            
            # Collect consecutive table-like lines
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    # Empty line might end table
                    if len(table_lines) > 2:  # At least header + 1 row
                        break
                    i += 1
                    continue
                
                if '\t' in next_line or '|' in next_line:
                    table_lines.append(next_line)
                    i += 1
                else:
                    # Line doesn't look like table row
                    if len(table_lines) > 2:
                        break
                    else:
                        # Not a table, reset
                        table_lines = [next_line]
                        table_start = i
                        i += 1
            
            if len(table_lines) > 2:
                table_text = '\n'.join(table_lines)
                tables.append((table_text, table_start, i))
        else:
            i += 1
    
    return tables


def convert_table_to_text(table_text: str) -> str:
    """
    Convert table text to structured text format
    """
    try:
        # Try to parse as tab-separated
        if '\t' in table_text:
            df = pd.read_csv(StringIO(table_text), sep='\t')
        else:
            # Try pipe-separated
            df = pd.read_csv(StringIO(table_text), sep='|')
        
        # Convert to readable text
        text_output = "TABLE:\n"
        text_output += df.to_string(index=False)
        text_output += "\n"
        
        return text_output
    except:
        # If parsing fails, return original text with marker
        return f"[TABLE DATA]\n{table_text}\n[/TABLE DATA]\n"


def chunk_documents_with_tables(documents):
    """
    Chunk documents with table-aware processing
    Tables are kept intact and not split across chunks
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        
        # Extract tables
        tables = extract_tables_from_text(text)
        
        if not tables:
            # No tables, use regular chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
            continue
        
        # Process text with tables
        processed_text = text
        offset = 0
        
        for table_text, start_pos, end_pos in tables:
            # Convert table to structured text
            structured_table = convert_table_to_text(table_text)
            
            # Replace table in text
            # Find the actual position in the original text
            lines = text.split('\n')
            table_lines = table_text.split('\n')
            
            # Reconstruct text with table marker
            before_table = '\n'.join(lines[:start_pos])
            after_table = '\n'.join(lines[end_pos:])
            
            # Add table as a single chunk
            table_chunk = f"{before_table}\n{structured_table}\n{after_table}"
            
            # Create a separate chunk for the table
            table_doc = {
                "page_content": structured_table,
                "metadata": {**metadata, "type": "table"}
            }
            all_chunks.append(table_doc)
            
            # Process the rest of the text (excluding table)
            remaining_text = before_table + "\n" + after_table
            if remaining_text.strip():
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ".", " "]
                )
                remaining_doc = type('obj', (object,), {
                    'page_content': remaining_text,
                    'metadata': {**metadata, "type": "text"}
                })
                chunks = splitter.split_documents([remaining_doc])
                all_chunks.extend(chunks)
    
    print(f"Total chunks created (table-aware): {len(all_chunks)}")
    return all_chunks


def load_pdfs():
    documents = []
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in knowledge_base/ folder")
        return []

    for pdf_file in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_DIR, pdf_file)
        print(f"Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    """
    Legacy chunking function (non-table-aware)
    Use chunk_documents_with_tables for table-aware processing
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def ingest(table_aware: bool = True):
    """
    Ingest PDFs with optional table-aware chunking
    
    Args:
        table_aware: If True, use table-aware chunking (default: True)
    """
    documents = load_pdfs()
    
    if table_aware:
        chunks = chunk_documents_with_tables(documents)
    else:
        chunks = chunk_documents(documents)
    
    return chunks