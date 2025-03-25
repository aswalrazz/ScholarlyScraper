"""
Web scraper utility functions for enhancing scholarly data.
This module provides functionality to extract text content from web pages
using the trafilatura library to enrich publication data with additional content.
"""

import trafilatura
import pandas as pd
import time
import re
from urllib.parse import urlparse

def get_website_text_content(url: str) -> str:
    """
    Extract the main text content from a website.
    
    Args:
        url (str): URL of the website to scrape
        
    Returns:
        str: Extracted text content or empty string if extraction fails
    """
    try:
        # Add a small delay to prevent too many requests
        time.sleep(0.5)
        
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            # Extract text content
            text = trafilatura.extract(downloaded)
            return text if text else ""
        else:
            return ""
    except Exception as e:
        print(f"Error fetching content from {url}: {str(e)}")
        return ""

def extract_doi_from_url(url: str) -> str:
    """
    Extract DOI from a URL if present.
    
    Args:
        url (str): URL potentially containing a DOI
        
    Returns:
        str: Extracted DOI or empty string
    """
    # Check if it's a DOI URL
    if "doi.org" in url:
        parsed = urlparse(url)
        path = parsed.path
        if path.startswith("/"):
            # Remove leading slash
            path = path[1:]
        return path
    
    # Try to find DOI pattern in URL
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    match = re.search(doi_pattern, url)
    if match:
        return match.group(0)
    
    return ""

def enrich_publication_data(df: pd.DataFrame, max_items: int = None) -> pd.DataFrame:
    """
    Enrich publication data by scraping additional content from publication URLs.
    
    Args:
        df (pd.DataFrame): DataFrame containing publication data with 'doi' column
        max_items (int, optional): Maximum number of items to process. Defaults to None (all items).
        
    Returns:
        pd.DataFrame: Enriched DataFrame with additional content
    """
    if 'doi' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    enriched_df = df.copy()
    
    # Add abstract column if it doesn't exist
    if 'abstract' not in enriched_df.columns:
        enriched_df['abstract'] = ""
    
    # Add full_text column for scraped content
    enriched_df['full_text'] = ""
    
    # Process items up to max_items if specified
    process_items = df.shape[0] if max_items is None else min(max_items, df.shape[0])
    
    for i in range(process_items):
        if pd.isna(df.iloc[i]['doi']) or not df.iloc[i]['doi']:
            continue
            
        # Format DOI URL if not already formatted
        doi = df.iloc[i]['doi']
        if not doi.startswith("http"):
            doi_url = f"https://doi.org/{doi}"
        else:
            doi_url = doi
            
        # If abstract is empty, try to get it from the publication URL
        if pd.isna(df.iloc[i]['abstract']) or not df.iloc[i]['abstract']:
            content = get_website_text_content(doi_url)
            
            if content:
                # Try to extract abstract (first 500 characters as a simple heuristic)
                abstract = content[:500] if len(content) > 500 else content
                enriched_df.iloc[i, enriched_df.columns.get_loc('abstract')] = abstract
                
                # Store full text for potential further analysis
                enriched_df.iloc[i, enriched_df.columns.get_loc('full_text')] = content
    
    return enriched_df

def find_related_publications(url: str, max_links: int = 5) -> list:
    """
    Find related publications by extracting links from a publication page.
    
    Args:
        url (str): URL of the publication page
        max_links (int, optional): Maximum number of links to return. Defaults to 5.
        
    Returns:
        list: List of related publication URLs
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return []
            
        # Extract links
        links = []
        
        # Use regular expressions to find DOI links
        doi_pattern = r'https://doi\.org/10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
        matches = re.findall(doi_pattern, downloaded.decode('utf-8', errors='ignore'))
        
        # Deduplicate and filter out the original URL
        unique_links = set([link for link in matches if link != url])
        
        return list(unique_links)[:max_links]
    except Exception as e:
        print(f"Error finding related publications from {url}: {str(e)}")
        return []