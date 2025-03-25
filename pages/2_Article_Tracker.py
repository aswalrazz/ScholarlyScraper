import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(page_title="Article Tracker", page_icon="📄", layout="wide")

# Page title
st.title("Article Tracker")
st.markdown("Track and analyze individual articles or groups of articles")

# Check if search has been performed and data exists
if 'search_performed' not in st.session_state or not st.session_state.search_performed:
    st.info("Please perform a search on the main page first to track articles.")
elif 'search_results' not in st.session_state or st.session_state.search_results is None or st.session_state.search_results.empty:
    st.warning("No articles available for tracking. Please search for scholarly content on the main page.")
else:
    # Get data from session state
    df = st.session_state.search_results
    
    # Sidebar for article selection
    with st.sidebar:
        st.header("Article Selection")
        
        selection_method = st.radio(
            "Selection Method",
            ["Individual Article", "Multiple Articles"]
        )
        
        if selection_method == "Individual Article":
            # Dropdown to select a specific article
            article_titles = df['title'].tolist()
            selected_article = st.selectbox(
                "Select an article to track",
                options=article_titles
            )
            
            # Filter dataframe to get just the selected article
            selected_df = df[df['title'] == selected_article]
            
        else:  # Multiple Articles
            # Allow filtering by metadata
            st.subheader("Filter Articles")
            
            # Filter by year range
            min_year = int(df['publication_date'].dt.year.min())
            max_year = int(df['publication_date'].dt.year.max())
            year_range = st.slider(
                "Publication Year",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Filter by journal if available
            if 'journal' in df.columns and not df['journal'].isna().all():
                journals = ['All'] + sorted(df['journal'].unique().tolist())
                selected_journal = st.selectbox(
                    "Journal",
                    options=journals
                )
            else:
                selected_journal = 'All'
                
            # Filter by author
            if 'authors' in df.columns:
                # Extract all authors and remove duplicates
                all_authors = []
                for authors_str in df['authors'].dropna():
                    authors_list = [author.strip() for author in authors_str.split(',')]
                    all_authors.extend(authors_list)
                
                unique_authors = ['All'] + sorted(list(set(all_authors)))
                selected_author = st.selectbox(
                    "Author",
                    options=unique_authors
                )
            else:
                selected_author = 'All'
            
            # Apply filters to get selected articles
            selected_df = df[
                (df['publication_date'].dt.year >= year_range[0]) &
                (df['publication_date'].dt.year <= year_range[1])
            ]
            
            if selected_journal != 'All':
                selected_df = selected_df[selected_df['journal'] == selected_journal]
                
            if selected_author != 'All':
                selected_df = selected_df[selected_df['authors'].str.contains(selected_author, na=False)]
    
    # Display article(s) information
    if not selected_df.empty:
        if len(selected_df) == 1:
            # Single article view
            article = selected_df.iloc[0]
            
            st.subheader(article['title'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Authors:** {article['authors']}")
                st.markdown(f"**Publication Date:** {article['publication_date'].strftime('%B %d, %Y') if pd.notna(article['publication_date']) else 'Unknown'}")
                st.markdown(f"**Journal/Source:** {article['journal'] if 'journal' in article and pd.notna(article['journal']) else 'Unknown'}")
                
            with col2:
                st.markdown(f"**DOI:** [{article['doi']}](https://doi.org/{article['doi']})")
                st.markdown(f"**Citations:** {int(article['citations'])}")
                st.markdown(f"**Type:** {article['type'] if 'type' in article and pd.notna(article['type']) else 'Unknown'}")
            
            # Abstract if available
            if 'abstract' in article and pd.notna(article['abstract']) and article['abstract'] != '':
                with st.expander("Abstract", expanded=True):
                    st.write(article['abstract'])
            
            # Keywords if available
            if 'keywords' in article and pd.notna(article['keywords']) and article['keywords'] != '':
                with st.expander("Keywords/Concepts"):
                    keywords = [kw.strip() for kw in article['keywords'].split(',')]
                    st.write(", ".join(keywords))
            
            # Citation metrics and visualizations
            st.subheader("Impact Metrics")
            
            citations = int(article['citations'])
            years_since_pub = datetime.now().year - article['publication_date'].year if pd.notna(article['publication_date']) else 0
            citations_per_year = citations / max(1, years_since_pub)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Citations", citations)
            metric_col2.metric("Years Since Publication", years_since_pub)
            metric_col3.metric("Citations Per Year", f"{citations_per_year:.2f}")
            
            # Citation comparison (if there are other articles in the dataset)
            if len(df) > 1:
                st.subheader("Comparative Impact")
                
                # Compare with average citation rate for same year
                same_year_df = df[df['publication_date'].dt.year == article['publication_date'].year]
                avg_citations = same_year_df['citations'].mean()
                percentile = sum(df['citations'] <= citations) / len(df) * 100
                
                comp_col1, comp_col2 = st.columns(2)
                
                comp_col1.metric(
                    "Citations vs. Avg for Same Year", 
                    f"{citations:.0f}", 
                    f"{citations - avg_citations:.1f} ({(citations / max(1, avg_citations) - 1) * 100:.1f}%)"
                )
                comp_col2.metric("Citation Percentile", f"{percentile:.1f}%")
                
                # Chart showing citation comparison
                year_avg_citations = df.groupby(df['publication_date'].dt.year)['citations'].mean().reset_index()
                year_avg_citations.columns = ['year', 'avg_citations']
                
                fig = px.line(
                    year_avg_citations, 
                    x='year', 
                    y='avg_citations',
                    title='Average Citations by Publication Year',
                    labels={'year': 'Publication Year', 'avg_citations': 'Average Citations'}
                )
                
                # Add point for the current article
                fig.add_scatter(
                    x=[article['publication_date'].year], 
                    y=[citations],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name=f"Selected Article ({article['publication_date'].year}: {citations} citations)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Multiple articles view
            st.subheader(f"Tracking {len(selected_df)} Articles")
            
            # Basic statistics
            total_citations = selected_df['citations'].sum()
            avg_citations = selected_df['citations'].mean()
            max_citation = selected_df['citations'].max()
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Total Articles", len(selected_df))
            stat_col2.metric("Total Citations", f"{total_citations:.0f}")
            stat_col3.metric("Average Citations", f"{avg_citations:.2f}")
            stat_col4.metric("Maximum Citations", f"{max_citation:.0f}")
            
            # Publications over time
            st.subheader("Publications Over Time")
            pub_by_year = selected_df.groupby(selected_df['publication_date'].dt.year).size().reset_index(name='count')
            
            fig1 = px.bar(
                pub_by_year, 
                x='publication_date', 
                y='count',
                title='Publications by Year',
                labels={'publication_date': 'Year', 'count': 'Number of Publications'}
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Citations analysis
            st.subheader("Citation Analysis")
            
            # Group by year and get citation statistics
            citation_by_year = selected_df.groupby(selected_df['publication_date'].dt.year)['citations'].agg(['sum', 'mean', 'median', 'max']).reset_index()
            citation_by_year.columns = ['year', 'total_citations', 'mean_citations', 'median_citations', 'max_citations']
            
            citation_viz = st.radio(
                "Citation Visualization", 
                ["Total Citations", "Average Citations", "Citation Distribution"]
            )
            
            if citation_viz == "Total Citations":
                fig2 = px.bar(
                    citation_by_year,
                    x='year',
                    y='total_citations',
                    title='Total Citations by Publication Year',
                    labels={'year': 'Publication Year', 'total_citations': 'Total Citations'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            elif citation_viz == "Average Citations":
                fig2 = px.line(
                    citation_by_year,
                    x='year',
                    y=['mean_citations', 'median_citations', 'max_citations'],
                    title='Citation Statistics by Publication Year',
                    labels={'year': 'Publication Year', 'value': 'Citations', 'variable': 'Statistic'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            elif citation_viz == "Citation Distribution":
                fig2 = px.box(
                    selected_df,
                    x=selected_df['publication_date'].dt.year,
                    y='citations',
                    title='Citation Distribution by Publication Year',
                    labels={'x': 'Publication Year', 'citations': 'Number of Citations'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Top cited papers in the selection
            st.subheader("Most Cited Articles")
            top_papers = selected_df.sort_values('citations', ascending=False).head(10)
            st.dataframe(
                top_papers[['title', 'authors', 'publication_date', 'journal', 'citations', 'doi']],
                use_container_width=True,
                column_config={
                    "citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "doi": st.column_config.LinkColumn("DOI", display_text="View"),
                }
            )
    else:
        st.warning("No articles match the selected criteria. Please adjust your filters.")
