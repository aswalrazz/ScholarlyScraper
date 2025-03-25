import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_impact_metrics(df):
    """
    Calculate comprehensive impact metrics from scholarly data
    
    Args:
        df (pd.DataFrame): DataFrame with citation data
        
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
            'g_index': 0,
            'citation_percentiles': []
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
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    citation_percentiles = []
    
    for p in percentiles:
        value = np.percentile(df['citations'], p)
        citation_percentiles.append({
            'percentile': p,
            'value': value
        })
    
    return {
        'total_publications': total_publications,
        'total_citations': total_citations,
        'avg_citations': avg_citations,
        'h_index': h_index,
        'i10_index': i10_index,
        'g_index': g_index,
        'citation_percentiles': citation_percentiles
    }

def analyze_publications_by_time(df):
    """
    Analyze publication and citation patterns over time
    
    Args:
        df (pd.DataFrame): DataFrame with publication and citation data
        
    Returns:
        dict: Dictionary of time-based analyses
    """
    if df.empty or 'publication_date' not in df.columns:
        return {
            'publications_by_year': pd.DataFrame(),
            'citations_by_year': pd.DataFrame(),
            'citation_velocity': pd.DataFrame()
        }
    
    # Ensure date column is datetime
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    
    # Group by year
    df['year'] = df['publication_date'].dt.year
    
    # Publications by year
    publications_by_year = df.groupby('year').size().reset_index(name='count')
    
    # Citations by year
    citations_by_year = df.groupby('year')['citations'].agg(['sum', 'mean', 'median']).reset_index()
    
    # Calculate citation velocity
    # How quickly papers accumulate citations relative to their age
    current_year = datetime.now().year
    
    df['years_since_publication'] = current_year - df['year']
    df['citation_velocity'] = df['citations'] / df['years_since_publication'].clip(lower=1)
    
    citation_velocity = df.groupby('year')['citation_velocity'].mean().reset_index()
    
    return {
        'publications_by_year': publications_by_year,
        'citations_by_year': citations_by_year,
        'citation_velocity': citation_velocity
    }

def analyze_authors(df):
    """
    Analyze author impact and collaboration patterns
    
    Args:
        df (pd.DataFrame): DataFrame with author and citation data
        
    Returns:
        dict: Dictionary of author-based analyses
    """
    if df.empty or 'authors' not in df.columns:
        return {
            'author_impact': pd.DataFrame(),
            'author_collaborations': pd.DataFrame()
        }
    
    # Extract author information
    all_authors = []
    author_papers = {}
    
    for _, row in df.iterrows():
        if pd.notna(row['authors']) and row['authors'] != '':
            paper_authors = [author.strip() for author in row['authors'].split(',')]
            
            for author in paper_authors:
                if author:
                    all_authors.append(author)
                    
                    if author not in author_papers:
                        author_papers[author] = []
                    
                    author_papers[author].append(row)
    
    # Calculate impact metrics for each author
    author_metrics = []
    
    for author, papers in author_papers.items():
        if len(papers) < 2:  # Skip authors with only one paper
            continue
            
        paper_df = pd.DataFrame(papers)
        citations = paper_df['citations'].sum()
        avg_citations = paper_df['citations'].mean()
        paper_count = len(papers)
        
        # Calculate h-index for author
        citation_counts = paper_df['citations'].sort_values(ascending=False).values
        h_index = sum(citation_counts >= np.arange(1, len(citation_counts) + 1))
        
        author_metrics.append({
            'author': author,
            'paper_count': paper_count,
            'citations': citations,
            'avg_citations': avg_citations,
            'h_index': h_index
        })
    
    author_impact = pd.DataFrame(author_metrics)
    
    # Analyze author collaborations
    # Only include authors with multiple papers
    significant_authors = author_impact[author_impact['paper_count'] >= 2]['author'].tolist()
    
    # Create author collaboration matrix
    collaborations = np.zeros((len(significant_authors), len(significant_authors)))
    author_indices = {author: i for i, author in enumerate(significant_authors)}
    
    for _, row in df.iterrows():
        if pd.notna(row['authors']) and row['authors'] != '':
            paper_authors = [
                author.strip() 
                for author in row['authors'].split(',') 
                if author.strip() in significant_authors
            ]
            
            for i, author1 in enumerate(paper_authors):
                idx1 = author_indices.get(author1)
                if idx1 is not None:
                    for author2 in paper_authors[i+1:]:
                        idx2 = author_indices.get(author2)
                        if idx2 is not None:
                            collaborations[idx1, idx2] += 1
                            collaborations[idx2, idx1] += 1
    
    author_collaborations = pd.DataFrame(
        collaborations,
        index=significant_authors,
        columns=significant_authors
    )
    
    return {
        'author_impact': author_impact,
        'author_collaborations': author_collaborations
    }

def analyze_keywords(df):
    """
    Analyze keywords/topics and their impact
    
    Args:
        df (pd.DataFrame): DataFrame with keyword and citation data
        
    Returns:
        dict: Dictionary of keyword-based analyses
    """
    if df.empty:
        return {
            'keyword_frequency': pd.DataFrame(),
            'keyword_impact': pd.DataFrame(),
            'keyword_trends': pd.DataFrame()
        }
    
    # Extract keywords
    has_keywords = 'keywords' in df.columns and not df['keywords'].isna().all()
    
    if has_keywords:
        # Use explicit keywords
        all_keywords = []
        keyword_papers = {}
        
        for _, row in df.iterrows():
            if pd.notna(row['keywords']) and row['keywords'] != '':
                paper_keywords = [kw.strip() for kw in row['keywords'].split(',')]
                
                for keyword in paper_keywords:
                    if keyword:
                        all_keywords.append(keyword)
                        
                        if keyword not in keyword_papers:
                            keyword_papers[keyword] = []
                        
                        keyword_papers[keyword].append(row)
        
        # Calculate frequency
        keyword_frequency = pd.Series(all_keywords).value_counts().reset_index()
        keyword_frequency.columns = ['keyword', 'frequency']
        
        # Calculate impact metrics for each keyword
        keyword_metrics = []
        
        for keyword, papers in keyword_papers.items():
            if len(papers) < 3:  # Skip keywords with too few papers
                continue
                
            paper_df = pd.DataFrame(papers)
            citations = paper_df['citations'].sum()
            avg_citations = paper_df['citations'].mean()
            paper_count = len(papers)
            
            keyword_metrics.append({
                'keyword': keyword,
                'paper_count': paper_count,
                'total_citations': citations,
                'avg_citations': avg_citations
            })
        
        keyword_impact = pd.DataFrame(keyword_metrics)
        
        # Calculate keyword trends over time
        keyword_trends = []
        
        # Only analyze trends for top keywords
        top_keywords = keyword_frequency.head(20)['keyword'].tolist()
        
        for keyword in top_keywords:
            if keyword in keyword_papers:
                papers = keyword_papers[keyword]
                paper_df = pd.DataFrame(papers)
                
                if 'publication_date' in paper_df.columns:
                    paper_df['year'] = pd.to_datetime(paper_df['publication_date']).dt.year
                    yearly_counts = paper_df.groupby('year').size().reset_index(name='count')
                    
                    for _, row in yearly_counts.iterrows():
                        keyword_trends.append({
                            'keyword': keyword,
                            'year': row['year'],
                            'count': row['count']
                        })
        
        keyword_trends = pd.DataFrame(keyword_trends)
    else:
        # Extract terms from titles
        vectorizer = CountVectorizer(stop_words='english', min_df=3, ngram_range=(1, 2))
        
        # Check if title column exists
        if 'title' in df.columns and not df['title'].isna().all():
            title_vectors = vectorizer.fit_transform(df['title'].fillna(''))
            terms = vectorizer.get_feature_names_out()
            term_freq = np.sum(title_vectors.toarray(), axis=0)
            
            keyword_frequency = pd.DataFrame({
                'keyword': terms,
                'frequency': term_freq
            }).sort_values('frequency', ascending=False)
            
            # Calculate impact for top terms
            keyword_impact = []
            keyword_trends = []
            
            # Mock data since we can't easily associate titles with keywords
            keyword_impact = pd.DataFrame()
            keyword_trends = pd.DataFrame()
        else:
            keyword_frequency = pd.DataFrame()
            keyword_impact = pd.DataFrame()
            keyword_trends = pd.DataFrame()
    
    return {
        'keyword_frequency': keyword_frequency,
        'keyword_impact': keyword_impact,
        'keyword_trends': keyword_trends
    }

def find_similar_papers(df, target_paper_index, n=5):
    """
    Find papers similar to a target paper based on content
    
    Args:
        df (pd.DataFrame): DataFrame with paper data
        target_paper_index (int): Index of the target paper
        n (int): Number of similar papers to return
        
    Returns:
        pd.DataFrame: DataFrame with similar papers
    """
    if df.empty or len(df) <= 1:
        return pd.DataFrame()
    
    target_paper = df.iloc[target_paper_index]
    
    # Check which features we can use for comparison
    features = []
    
    if 'abstract' in df.columns and not df['abstract'].isna().all():
        features.append('abstract')
    
    if 'title' in df.columns and not df['title'].isna().all():
        features.append('title')
    
    if 'keywords' in df.columns and not df['keywords'].isna().all():
        features.append('keywords')
    
    if not features:
        return pd.DataFrame()
    
    # Prepare content for comparison
    documents = []
    
    for _, row in df.iterrows():
        content = []
        
        for feature in features:
            if pd.notna(row[feature]):
                content.append(str(row[feature]))
        
        documents.append(' '.join(content))
    
    # Calculate similarities
    vectorizer = CountVectorizer(stop_words='english')
    content_vectors = vectorizer.fit_transform(documents)
    
    # Calculate similarities
    similarities = cosine_similarity(content_vectors)
    
    # Get indices of similar papers (excluding the target paper itself)
    similar_indices = np.argsort(similarities[target_paper_index])[::-1][1:n+1]
    
    return df.iloc[similar_indices].copy()
