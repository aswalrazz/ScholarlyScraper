import pandas as pd
import numpy as np
from datetime import datetime
import re

def process_impact_data(openalex_data, crossref_data):
    """
    Process and combine data from OpenAlex and Crossref APIs
    
    Args:
        openalex_data (dict): Data from OpenAlex API
        crossref_data (dict): Data from Crossref API
        
    Returns:
        pd.DataFrame: Combined and processed data or None if both APIs return no data
    """
    # Check if both APIs returned None or empty data
    if (openalex_data is None or not openalex_data) and (crossref_data is None or not crossref_data):
        print("Both APIs returned no data.")
        return pd.DataFrame()
    
    openalex_df = process_openalex_data(openalex_data)
    crossref_df = process_crossref_data(crossref_data)
    
    # Combine data from both sources, preferring OpenAlex when duplicates exist
    if openalex_df is not None and not openalex_df.empty:
        if crossref_df is not None and not crossref_df.empty:
            try:
                # Merge on DOI where available
                combined_df = pd.merge(
                    openalex_df, 
                    crossref_df, 
                    on='doi', 
                    how='outer', 
                    suffixes=('_openalex', '_crossref')
                )
                
                # Process merged data to create uniform columns
                combined_df = process_merged_data(combined_df)
            except Exception as e:
                print(f"Error merging dataframes: {e}")
                # If merge fails, use OpenAlex data as fallback
                combined_df = openalex_df
        else:
            combined_df = openalex_df
    elif crossref_df is not None and not crossref_df.empty:
        combined_df = crossref_df
    else:
        # Return empty DataFrame if no data from either source
        return pd.DataFrame()
    
    try:
        # Fill NaN values and calculate additional metrics
        combined_df = enrich_data(combined_df)
    except Exception as e:
        print(f"Error enriching data: {e}")
        # If enrichment fails, return the combined dataframe as is
        pass
    
    return combined_df

def process_openalex_data(data):
    """
    Process data from OpenAlex API into a DataFrame
    
    Args:
        data (dict): OpenAlex API response
        
    Returns:
        pd.DataFrame: Processed data
    """
    if not data:
        print("OpenAlex data is None or empty")
        return None
    
    # Check if it's a single work or a list of works
    if 'results' in data:
        items = data.get('results', [])
    else:
        # Single work
        items = [data]
    
    if not items:
        print("No items found in OpenAlex data")
        return None
    
    records = []
    
    for item in items:
        try:
            if not item:
                continue
                
            # Extract comprehensive metadata with better error handling
            publication_year = None
            if item.get('publication_date'):
                try:
                    publication_year = int(item.get('publication_date').split('-')[0])
                except:
                    pass
                    
            record = {
                'title': item.get('title', ''),
                'doi': item.get('doi', '').replace('https://doi.org/', ''),
                'publication_date': item.get('publication_date'),
                'year': publication_year,
                'type': item.get('type', 'article'),
                'citations': int(item.get('cited_by_count', 0)),  # Direct citations
                'cited_by': int(item.get('cited_by_count', 0)),  # Also store as cited_by for clarity
                'related_count': item.get('related_count', 0) if isinstance(item.get('related_count'), int) else 0,
                'abstract': item.get('abstract', ''),
                'source': item.get('source', {}).get('display_name', '') if isinstance(item.get('source'), dict) else '',
                # Open access status - transformed to open/closed format
                'open_access_status': 'open' if (
                    isinstance(item.get('open_access'), dict) and 
                    item.get('open_access', {}).get('is_oa', False)
                ) else 'closed'
            }
            
            # Extract citation percentiles if available
            if 'citation_percentiles' in item and isinstance(item['citation_percentiles'], dict):
                record['citation_percentile'] = item['citation_percentiles'].get('year_subfield', 0)
            else:
                record['citation_percentile'] = 0
                
            # Try to extract FWCI (Field-Weighted Citation Impact)
            record['fwci'] = 0
            if 'counts_by_year' in item and isinstance(item['counts_by_year'], list) and item['counts_by_year']:
                for year_data in item['counts_by_year']:
                    if isinstance(year_data, dict) and 'fwci' in year_data:
                        record['fwci'] = year_data.get('fwci', 0)
                        break
                        
            # Extract fields, subfields, and domains from concepts
            topic_concepts = []
            subfields = []
            fields = []
            domains = []
            
            # Process concepts to extract subject areas
            if 'concepts' in item and isinstance(item['concepts'], list):
                for concept in item['concepts']:
                    if isinstance(concept, dict):
                        # High-score concepts are likely topics
                        if concept.get('score', 0) >= 0.7 and 'display_name' in concept:
                            topic_concepts.append(concept['display_name'])
                        
                        # Extract classification level information if available
                        if 'level' in concept and 'display_name' in concept:
                            level = concept.get('level', 0)
                            name = concept.get('display_name', '')
                            
                            if level == 0 and name: # Domain
                                domains.append(name)
                            elif level == 1 and name: # Field
                                fields.append(name)
                            elif level == 2 and name: # Subfield
                                subfields.append(name)
            
            record['topic'] = topic_concepts[0] if topic_concepts else ''
            record['subfield'] = subfields[0] if subfields else 'General'
            record['field'] = fields[0] if fields else 'General'
            record['domain'] = domains[0] if domains else 'General'
            
            # Extract authors and institutions
            authors = []
            institutions = []
            
            for author in item.get('authorships', []):
                if 'author' in author and isinstance(author['author'], dict):
                    author_name = author['author'].get('display_name', '')
                    if author_name:
                        authors.append(author_name)
                    
                    # Extract author institution
                    if 'institutions' in author and author['institutions']:
                        if isinstance(author['institutions'], list):
                            for institution in author['institutions']:
                                if isinstance(institution, dict) and 'display_name' in institution:
                                    inst_name = institution['display_name']
                                    if inst_name:
                                        institutions.append(inst_name)
            
            record['authors'] = ', '.join(authors) if authors else ''
            
            # Ensure institutions are properly joined and not empty
            if institutions:
                record['institutions'] = ', '.join(list(set(institutions)))
            else:
                record['institutions'] = 'Not specified'
            
            # Extract journal/source with better error handling
            if ('primary_location' in item and 
                isinstance(item['primary_location'], dict) and 
                'source' in item['primary_location'] and 
                isinstance(item['primary_location']['source'], dict)):
                
                record['journal'] = item['primary_location']['source'].get('display_name', '')
                record['publisher'] = item['primary_location']['source'].get('host_organization_name', '')
            else:
                record['journal'] = item.get('host_venue', {}).get('display_name', '') if isinstance(item.get('host_venue'), dict) else ''
                record['publisher'] = item.get('host_venue', {}).get('publisher', '') if isinstance(item.get('host_venue'), dict) else ''
            
            # Extract concepts/keywords
            concepts = []
            for concept in item.get('concepts', []):
                if isinstance(concept, dict) and concept.get('score', 0) >= 0.5:  # Only include relevant concepts
                    concept_name = concept.get('display_name', '')
                    if concept_name:
                        concepts.append(concept_name)
            record['keywords'] = ', '.join(concepts) if concepts else ''
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing OpenAlex item: {e}")
            continue
    
    if not records:
        print("No records extracted from OpenAlex data")
        return None
        
    df = pd.DataFrame(records)
    print(f"Processed {len(df)} records from OpenAlex data")
    
    # Debug information
    print(f"Columns in OpenAlex dataframe: {df.columns.tolist()}")
    print(f"Citations available: {df['citations'].sum()} total citations across {len(df)} records")
    print(f"Institutions data available: {df['institutions'].notna().sum()} records have institution data")
    
    return df

def process_crossref_data(data):
    """
    Process data from Crossref API into a DataFrame
    
    Args:
        data (dict): Crossref API response
        
    Returns:
        pd.DataFrame: Processed data
    """
    if not data:
        return None
    
    # Check if it's a single work or a list of works
    if 'message' in data and 'items' in data['message']:
        items = data['message'].get('items', [])
    elif 'message' in data:
        # Single work
        items = [data['message']]
    else:
        return None
    
    if not items:
        return None
    
    records = []
    
    for item in items:
        if not item:
            continue
            
        # Extract basic metadata
        record = {
            'title': item.get('title', [''])[0] if isinstance(item.get('title', []), list) else '',
            'doi': item.get('DOI', ''),
            'type': item.get('type', '')
        }
        
        # Extract publication date
        if 'published-print' in item and 'date-parts' in item['published-print']:
            date_parts = item['published-print']['date-parts'][0]
            if len(date_parts) >= 3:
                record['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
            elif len(date_parts) >= 2:
                record['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-01"
            elif len(date_parts) >= 1:
                record['publication_date'] = f"{date_parts[0]}-01-01"
        elif 'created' in item and 'date-parts' in item['created']:
            date_parts = item['created']['date-parts'][0]
            if len(date_parts) >= 3:
                record['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
            elif len(date_parts) >= 2:
                record['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-01"
            elif len(date_parts) >= 1:
                record['publication_date'] = f"{date_parts[0]}-01-01"
        else:
            record['publication_date'] = None
        
        # Extract authors
        authors = []
        for author in item.get('author', []):
            if 'given' in author and 'family' in author:
                authors.append(f"{author['given']} {author['family']}")
            elif 'family' in author:
                authors.append(author['family'])
        record['authors'] = ', '.join(authors) if authors else ''
        
        # Extract journal/source
        record['journal'] = item.get('container-title', [''])[0] if isinstance(item.get('container-title', []), list) else ''
        record['publisher'] = item.get('publisher', '')
        
        # Extract references count as proxy for citations
        record['references_count'] = item.get('references-count', 0)
        
        # Extract subject/keywords
        subjects = item.get('subject', [])
        record['keywords'] = ', '.join(subjects) if subjects else ''
        
        # Extract abstract
        record['abstract'] = item.get('abstract', '')
        
        records.append(record)
    
    if not records:
        return None
        
    return pd.DataFrame(records)

def process_merged_data(df):
    """
    Process merged data from OpenAlex and Crossref
    
    Args:
        df (pd.DataFrame): Merged DataFrame
        
    Returns:
        pd.DataFrame: Processed DataFrame with uniform columns
    """
    print(f"Processing merged dataframe with columns: {df.columns.tolist()}")
    
    # Create uniform columns, preferring OpenAlex data where available
    uniform_df = pd.DataFrame()
    
    try:
        # Title
        if 'title_openalex' in df.columns and 'title_crossref' in df.columns:
            uniform_df['title'] = df['title_openalex'].fillna(df['title_crossref'])
        elif 'title_openalex' in df.columns:
            uniform_df['title'] = df['title_openalex']
        elif 'title_crossref' in df.columns:
            uniform_df['title'] = df['title_crossref']
        elif 'title' in df.columns:
            uniform_df['title'] = df['title']
        else:
            uniform_df['title'] = "Untitled"
            print("Warning: No title column found in merged data")
        
        # DOI
        if 'doi' in df.columns:
            uniform_df['doi'] = df['doi']
        else:
            uniform_df['doi'] = ""
            print("Warning: No DOI column found in merged data")
        
        # Publication date
        if 'publication_date_openalex' in df.columns and 'publication_date_crossref' in df.columns:
            uniform_df['publication_date'] = df['publication_date_openalex'].fillna(df['publication_date_crossref'])
        elif 'publication_date_openalex' in df.columns:
            uniform_df['publication_date'] = df['publication_date_openalex']
        elif 'publication_date_crossref' in df.columns:
            uniform_df['publication_date'] = df['publication_date_crossref']
        elif 'publication_date' in df.columns:
            uniform_df['publication_date'] = df['publication_date']
        
        # Type
        if 'type_openalex' in df.columns and 'type_crossref' in df.columns:
            uniform_df['type'] = df['type_openalex'].fillna(df['type_crossref'])
        elif 'type_openalex' in df.columns:
            uniform_df['type'] = df['type_openalex']
        elif 'type_crossref' in df.columns:
            uniform_df['type'] = df['type_crossref']
        elif 'type' in df.columns:
            uniform_df['type'] = df['type']
        
        # Authors
        if 'authors_openalex' in df.columns and 'authors_crossref' in df.columns:
            uniform_df['authors'] = df['authors_openalex'].fillna(df['authors_crossref'])
        elif 'authors_openalex' in df.columns:
            uniform_df['authors'] = df['authors_openalex']
        elif 'authors_crossref' in df.columns:
            uniform_df['authors'] = df['authors_crossref']
        elif 'authors' in df.columns:
            uniform_df['authors'] = df['authors']
        
        # Journal
        if 'journal_openalex' in df.columns and 'journal_crossref' in df.columns:
            uniform_df['journal'] = df['journal_openalex'].fillna(df['journal_crossref'])
        elif 'journal_openalex' in df.columns:
            uniform_df['journal'] = df['journal_openalex']
        elif 'journal_crossref' in df.columns:
            uniform_df['journal'] = df['journal_crossref']
        elif 'journal' in df.columns:
            uniform_df['journal'] = df['journal']
        
        # Publisher
        if 'publisher_openalex' in df.columns and 'publisher_crossref' in df.columns:
            uniform_df['publisher'] = df['publisher_openalex'].fillna(df['publisher_crossref'])
        elif 'publisher_openalex' in df.columns:
            uniform_df['publisher'] = df['publisher_openalex']
        elif 'publisher_crossref' in df.columns:
            uniform_df['publisher'] = df['publisher_crossref']
        elif 'publisher' in df.columns:
            uniform_df['publisher'] = df['publisher']
        
        # Keywords
        if 'keywords_openalex' in df.columns and 'keywords_crossref' in df.columns:
            uniform_df['keywords'] = df['keywords_openalex'].fillna(df['keywords_crossref'])
        elif 'keywords_openalex' in df.columns:
            uniform_df['keywords'] = df['keywords_openalex']
        elif 'keywords_crossref' in df.columns:
            uniform_df['keywords'] = df['keywords_crossref']
        elif 'keywords' in df.columns:
            uniform_df['keywords'] = df['keywords']
        
        # Abstract
        if 'abstract_openalex' in df.columns and 'abstract_crossref' in df.columns:
            uniform_df['abstract'] = df['abstract_openalex'].fillna(df['abstract_crossref'])
        elif 'abstract_openalex' in df.columns:
            uniform_df['abstract'] = df['abstract_openalex']
        elif 'abstract_crossref' in df.columns:
            uniform_df['abstract'] = df['abstract_crossref']
        elif 'abstract' in df.columns:
            uniform_df['abstract'] = df['abstract']
        
        # Citations - prefer OpenAlex citation count, fallback to Crossref references count
        if 'citations' in df.columns and 'references_count' in df.columns:
            uniform_df['citations'] = df['citations'].fillna(df['references_count'])
        elif 'citations' in df.columns:
            uniform_df['citations'] = df['citations']
        elif 'references_count' in df.columns:
            uniform_df['citations'] = df['references_count']
        else:
            uniform_df['citations'] = 0
            print("Warning: No citation data found in merged dataframe")
        
        # Open access status
        if 'open_access_status' in df.columns:
            uniform_df['open_access_status'] = df['open_access_status']
        elif 'is_open_access' in df.columns:
            # Convert boolean to string format
            uniform_df['open_access_status'] = df['is_open_access'].apply(lambda x: 'open' if x else 'closed')
        else:
            uniform_df['open_access_status'] = 'closed'
        
        # Institutions
        if 'institutions' in df.columns:
            uniform_df['institutions'] = df['institutions']
        elif 'institutions_openalex' in df.columns:
            uniform_df['institutions'] = df['institutions_openalex']
        else:
            uniform_df['institutions'] = 'Not specified'
            print("Warning: No institution data found in merged dataframe")
            
        # Add the new fields
        # Year
        if 'year' in df.columns:
            uniform_df['year'] = df['year']
        else:
            # Try to extract from publication date
            try:
                if 'publication_date' in uniform_df.columns:
                    uniform_df['year'] = pd.to_datetime(uniform_df['publication_date']).dt.year
                else:
                    uniform_df['year'] = None
            except Exception as e:
                print(f"Error extracting publication year: {e}")
                uniform_df['year'] = None
                
        # Source - conference/journal
        if 'source' in df.columns:
            uniform_df['source'] = df['source']
        elif 'journal' in uniform_df.columns:
            uniform_df['source'] = uniform_df['journal']
        else:
            uniform_df['source'] = ''
        
        # Cited by (same as citations)
        if 'cited_by' in df.columns:
            uniform_df['cited_by'] = df['cited_by']
        elif 'citations' in uniform_df.columns:
            uniform_df['cited_by'] = uniform_df['citations']
        else:
            uniform_df['cited_by'] = 0
            
        # Related count
        if 'related_count' in df.columns:
            uniform_df['related_count'] = df['related_count']
        else:
            uniform_df['related_count'] = 0
            
        # FWCI - Field-Weighted Citation Impact
        if 'fwci' in df.columns:
            uniform_df['fwci'] = df['fwci']
        else:
            uniform_df['fwci'] = 0
            
        # Citation percentile
        if 'citation_percentile' in df.columns:
            uniform_df['citation_percentile'] = df['citation_percentile']
        else:
            uniform_df['citation_percentile'] = 0
            
        # Topic, Subfield, Field, Domain
        for field in ['topic', 'subfield', 'field', 'domain']:
            if field in df.columns:
                uniform_df[field] = df[field]
            else:
                uniform_df[field] = ''
        
        print(f"Uniform dataframe contains {len(uniform_df)} rows with columns: {uniform_df.columns.tolist()}")
        return uniform_df
    
    except Exception as e:
        print(f"Error processing merged data: {e}")
        # Create a minimal dataset in case of errors
        if 'doi' in df.columns:
            uniform_df['doi'] = df['doi']
        if 'title' in df.columns:
            uniform_df['title'] = df['title']
        if df.empty:
            print("Warning: Returning empty dataframe due to processing error")
        return uniform_df

def enrich_data(df):
    """
    Enrich data with additional metrics and clean up
    
    Args:
        df (pd.DataFrame): DataFrame to enrich
        
    Returns:
        pd.DataFrame: Enriched DataFrame
    """
    if df.empty:
        print("DataFrame is empty, no enrichment performed")
        return df
    
    try:
        print(f"Starting data enrichment on dataframe with {len(df)} rows")
        print(f"Initial columns: {df.columns.tolist()}")
        
        # Ensure the citations column exists and is numeric
        if 'citations' not in df.columns:
            print("Citations column not found - adding with default values of 0")
            df['citations'] = 0
        else:
            # Convert to numeric, handling any non-numeric values
            print("Converting citations to numeric values")
            df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0)
        
        # Ensure institutions column exists
        if 'institutions' not in df.columns:
            print("Institutions column not found - adding default values")
            df['institutions'] = 'Not specified'
        
        # Convert publication date to datetime
        print("Converting publication dates to datetime format")
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        
        # Calculate years since publication
        current_year = datetime.now().year
        print(f"Calculating years since publication (current year: {current_year})")
        df['years_since_publication'] = current_year - df['publication_date'].dt.year
        
        # Calculate citations per year
        print("Calculating citations per year")
        df['citations_per_year'] = df.apply(
            lambda x: x['citations'] / max(x['years_since_publication'], 1) 
            if pd.notna(x['years_since_publication']) and x['years_since_publication'] > 0 else 0, 
            axis=1
        )
        
        # Clean text fields
        print("Cleaning text fields")
        text_columns = [
            'title', 'authors', 'journal', 'publisher', 'keywords', 
            'abstract', 'institutions', 'topic', 'subfield', 'field', 
            'domain', 'source', 'open_access_status'
        ]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x.strip() if x != 'nan' else '')
                # Count non-empty values
                non_empty = (df[col] != '').sum()
                print(f"Column '{col}': {non_empty}/{len(df)} non-empty values")
        
        # Convert numeric columns to appropriate types
        print("Converting numeric columns to appropriate types")
        numeric_columns = [
            'citations', 'citations_per_year', 'cited_by', 'related_count',
            'fwci', 'citation_percentile', 'year'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Round certain columns
                if col in ['citations_per_year', 'fwci']:
                    df[col] = df[col].round(3)
                elif col in ['citation_percentile']:
                    df[col] = df[col].round(2)
                elif col == 'year':
                    df[col] = df[col].astype('Int64')  # Integer that allows NaN
                else:
                    df[col] = df[col].astype(int)
        
        # Calculate h-index for each paper
        # This is a bit unusual as h-index is typically for authors/institutions,
        # but we'll calculate how each paper contributes to h-index
        print("Calculating paper-level h-index")
        try:
            # For each paper, we'll assign its h-index contribution as:
            # 1 if citations >= paper's rank among all papers, 0 otherwise
            citations_sorted = df['citations'].sort_values(ascending=False)
            citation_ranks = citations_sorted.reset_index().index + 1
            h_contributions = (citations_sorted.values >= citation_ranks).astype(int)
            
            # Map back to original dataframe
            h_index_map = dict(zip(citations_sorted.index, h_contributions))
            df['h_index_contribution'] = df.index.map(h_index_map).fillna(0).astype(int)
            
            # Calculate the overall h-index (should match calculate_metrics result)
            overall_h_index = h_contributions.sum()
            print(f"Overall h-index for dataset: {overall_h_index}")
        except Exception as e:
            print(f"Error calculating h-index contributions: {e}")
            df['h_index_contribution'] = 0
        
        # Drop rows with missing essential data
        initial_rows = len(df)
        df = df.dropna(subset=['title'])
        print(f"Dropped {initial_rows - len(df)} rows with missing titles")
        
        # Summarize citation statistics
        print(f"Citation statistics: Min={df['citations'].min()}, Max={df['citations'].max()}, Avg={df['citations'].mean():.2f}")
        
        return df
    except Exception as e:
        print(f"Error enriching data: {e}")
        # Return the original dataframe if enrichment fails
        return df

def calculate_metrics(df):
    """
    Calculate overall metrics for a dataset
    
    Args:
        df (pd.DataFrame): DataFrame with scholarly data
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    if df.empty:
        return {
            'total_publications': 0,
            'total_citations': 0,
            'avg_citations': 0,
            'h_index': 0,
            'i10_index': 0,
            'g_index': 0
        }
    
    # Ensure citation column exists and is numeric
    if 'citations' not in df.columns:
        df['citations'] = 0
    else:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0)
    
    # Basic metrics
    total_publications = len(df)
    total_citations = df['citations'].sum()
    avg_citations = total_citations / total_publications if total_publications > 0 else 0
    
    # Calculate h-index
    citation_counts = df['citations'].sort_values(ascending=False).values
    h_index = sum(citation_counts >= np.arange(1, len(citation_counts) + 1))
    
    # Calculate i10-index (number of publications with at least 10 citations)
    i10_index = sum(citation_counts >= 10)
    
    # Calculate g-index
    g_index = 0
    cumulative_citations = 0
    for i, count in enumerate(citation_counts, 1):
        cumulative_citations += count
        if cumulative_citations >= i*i:
            g_index = i
        else:
            break
    
    return {
        'total_publications': total_publications,
        'total_citations': total_citations,
        'avg_citations': avg_citations,
        'h_index': h_index,
        'i10_index': i10_index,
        'g_index': g_index
    }
