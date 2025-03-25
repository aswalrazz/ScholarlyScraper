import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import re

# Set page config
st.set_page_config(page_title="Impact Analysis", page_icon="🔍", layout="wide")

# Page title
st.title("Research Impact Analysis")
st.markdown("Comprehensive analysis of research impact across multiple dimensions")

# Check if search has been performed and data exists
if 'search_performed' not in st.session_state or not st.session_state.search_performed:
    st.info("Please perform a search on the main page first to analyze research impact.")
elif 'search_results' not in st.session_state or st.session_state.search_results is None or st.session_state.search_results.empty:
    st.warning("No data available for analysis. Please search for scholarly content on the main page.")
else:
    # Get data from session state
    df = st.session_state.search_results
    
    # Sidebar for analysis options
    with st.sidebar:
        st.header("Analysis Options")
        
        # Select analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ["Impact Overview", "Author Analysis", "Keyword/Topic Analysis", "Journal Analysis", "Temporal Analysis"]
        )
        
        # Filter options
        st.subheader("Filter Data")
        
        # Date range filter
        min_year = int(df['publication_date'].dt.year.min())
        max_year = int(datetime.now().year)
        year_range = st.slider(
            "Publication Years",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Citation threshold
        min_citations = st.number_input(
            "Minimum Citations",
            min_value=0,
            value=0
        )
    
    # Apply filters
    filtered_df = df[
        (df['publication_date'].dt.year >= year_range[0]) &
        (df['publication_date'].dt.year <= year_range[1]) &
        (df['citations'] >= min_citations)
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
    else:
        # Run the selected analysis
        if analysis_type == "Impact Overview":
            st.subheader("Research Impact Overview")
            
            # Calculate basic impact metrics
            total_publications = len(filtered_df)
            total_citations = filtered_df['citations'].sum()
            avg_citations = filtered_df['citations'].mean()
            
            # Calculate h-index from the filtered data
            citation_counts = filtered_df['citations'].sort_values(ascending=False).values
            h_index = sum(citation_counts >= np.arange(1, len(citation_counts) + 1))
            
            # Calculate i10-index (number of publications with at least 10 citations)
            i10_index = sum(citation_counts >= 10)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Publications", total_publications)
            col1.metric("Total Citations", int(total_citations))
            col2.metric("Average Citations", f"{avg_citations:.2f}")
            col2.metric("h-index", h_index)
            col3.metric("i10-index", i10_index)
            col3.metric("Citation Median", f"{filtered_df['citations'].median():.1f}")
            
            # Impact over time
            st.subheader("Impact Over Time")
            
            yearly_data = filtered_df.groupby(filtered_df['publication_date'].dt.year).agg({
                'title': 'count',
                'citations': 'sum'
            }).reset_index()
            yearly_data.columns = ['year', 'publications', 'citations']
            
            # Calculate cumulative publications and citations
            yearly_data['cum_publications'] = yearly_data['publications'].cumsum()
            yearly_data['cum_citations'] = yearly_data['citations'].cumsum()
            
            # Create a dual-axis chart
            fig = go.Figure()
            
            # Add publications line
            fig.add_trace(go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['publications'],
                name='Publications',
                line=dict(color='blue')
            ))
            
            # Add citations line
            fig.add_trace(go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['citations'],
                name='Citations',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            # Set layout
            fig.update_layout(
                title="Publications and Citations by Year",
                xaxis=dict(title="Year"),
                yaxis=dict(
                    title=dict(text="Publications", font=dict(color="blue")),
                    tickfont=dict(color="blue")
                ),
                yaxis2=dict(
                    title=dict(text="Citations", font=dict(color="red")),
                    tickfont=dict(color="red"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Citation distribution
            st.subheader("Citation Distribution")
            
            fig = px.histogram(
                filtered_df,
                x='citations',
                nbins=30,
                title='Distribution of Citations',
                labels={'citations': 'Number of Citations', 'count': 'Number of Publications'},
                log_x=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Most impactful publications
            st.subheader("Most Impactful Publications")
            
            top_papers = filtered_df.sort_values('citations', ascending=False).head(10)
            st.dataframe(
                top_papers[['title', 'authors', 'publication_date', 'journal', 'citations', 'doi']],
                use_container_width=True,
                column_config={
                    "citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "doi": st.column_config.LinkColumn("DOI", display_text="View"),
                }
            )
        
        elif analysis_type == "Author Analysis":
            st.subheader("Author Impact Analysis")
            
            # Extract authors from the data
            all_authors = []
            author_papers = {}
            author_citations = {}
            author_institutions = {}
            
            for _, row in filtered_df.iterrows():
                if pd.notna(row['authors']) and row['authors'] != '':
                    author_list = [author.strip() for author in row['authors'].split(',')]
                    
                    # Get institutions if available
                    institutions = row.get('institutions', '') if pd.notna(row.get('institutions', '')) else ''
                    
                    for author in author_list:
                        if author:
                            all_authors.append(author)
                            
                            # Track papers by author
                            if author not in author_papers:
                                author_papers[author] = []
                                author_institutions[author] = set()
                            
                            author_papers[author].append(row)
                            
                            # Add institutions for this author if available
                            if institutions:
                                inst_list = [inst.strip() for inst in institutions.split(',')]
                                for inst in inst_list:
                                    if inst:
                                        author_institutions[author].add(inst)
            
            # Count author frequencies
            author_counts = pd.Series(all_authors).value_counts()
            top_authors = author_counts.head(20)
            
            # Calculate citations per author
            for author, papers in author_papers.items():
                author_citations[author] = sum(paper['citations'] for paper in papers)
            
            # Create author statistics dataframe
            author_stats = []
            for author in top_authors.index:
                papers = author_papers[author]
                citations = author_citations[author]
                
                # Calculate h-index for the author
                citation_counts = [paper['citations'] for paper in papers]
                citation_counts.sort(reverse=True)
                h_index = sum(np.array(citation_counts) >= np.arange(1, len(citation_counts) + 1))
                
                # Get primary institutions for this author
                primary_institutions = ', '.join(list(author_institutions.get(author, [])))[:100]  # Limit length
                
                author_stats.append({
                    'Author': author,
                    'Institutions': primary_institutions,
                    'Publications': len(papers),
                    'Citations': citations,
                    'Avg Citations': citations / len(papers),
                    'h-index': h_index
                })
            
            author_stats_df = pd.DataFrame(author_stats)
            
            # Display author statistics
            st.subheader("Top Authors by Publication Count")
            
            # Sort by different metrics
            sort_by = st.radio(
                "Sort by",
                ["Publications", "Citations", "Avg Citations", "h-index"],
                horizontal=True
            )
            
            # Sort and display
            sorted_stats = author_stats_df.sort_values(sort_by, ascending=False)
            st.dataframe(
                sorted_stats,
                use_container_width=True,
                column_config={
                    "Institutions": st.column_config.TextColumn("Institutions"),
                    "Publications": st.column_config.NumberColumn(format="%d"),
                    "Citations": st.column_config.NumberColumn(format="%d"),
                    "Avg Citations": st.column_config.NumberColumn(format="%.2f"),
                    "h-index": st.column_config.NumberColumn(format="%d")
                }
            )
            
            # Visualize top authors
            st.subheader("Top Authors Visualization")
            
            # Get top 10 authors by the selected metric
            top_10_authors = sorted_stats.head(10)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_10_authors,
                y='Author',
                x=sort_by,
                title=f'Top 10 Authors by {sort_by}',
                orientation='h',
                text=sort_by,
                color=sort_by
            )
            
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Co-authorship analysis (simple version)
            st.subheader("Co-authorship Patterns")
            
            # Create simple co-authorship matrix for top authors
            top_author_names = sorted_stats.head(10)['Author'].tolist()
            coauthor_matrix = pd.DataFrame(0, index=top_author_names, columns=top_author_names)
            
            # Fill the matrix
            for _, row in filtered_df.iterrows():
                if pd.notna(row['authors']) and row['authors'] != '':
                    paper_authors = [author.strip() for author in row['authors'].split(',')]
                    paper_authors = [a for a in paper_authors if a in top_author_names]
                    
                    # Add co-authorship connections
                    for i, author1 in enumerate(paper_authors):
                        for author2 in paper_authors[i+1:]:
                            coauthor_matrix.loc[author1, author2] += 1
                            coauthor_matrix.loc[author2, author1] += 1
            
            # Display as heatmap
            fig = px.imshow(
                coauthor_matrix,
                text_auto=True,
                title='Co-authorship Matrix (Top 10 Authors)',
                labels=dict(x="Author", y="Author", color="Co-authored Papers"),
                color_continuous_scale="Blues"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Keyword/Topic Analysis":
            st.subheader("Keyword and Topic Analysis")
            
            # Extract keywords/topics from data
            all_keywords = []
            
            if 'keywords' in filtered_df.columns:
                # Use existing keywords if available
                for keywords in filtered_df['keywords'].dropna():
                    if keywords.strip():
                        keyword_list = [kw.strip() for kw in keywords.split(',')]
                        all_keywords.extend(keyword_list)
            else:
                # Fall back to extracting terms from titles
                st.warning("No explicit keywords found. Extracting common terms from titles.")
                for title in filtered_df['title'].dropna():
                    # Simple term extraction (not ideal but works as fallback)
                    # Remove common stopwords and punctuation
                    terms = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
                    stopwords = ['the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'not', 'have']
                    terms = [term for term in terms if term not in stopwords]
                    all_keywords.extend(terms)
            
            # Count keyword frequencies
            keyword_counts = pd.Series(all_keywords).value_counts()
            top_keywords = keyword_counts.head(20)
            
            # Display top keywords
            st.subheader("Most Common Keywords/Topics")
            
            fig = px.bar(
                x=top_keywords.index,
                y=top_keywords.values,
                title='Top 20 Keywords',
                labels={'x': 'Keyword', 'y': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Keyword co-occurrence (simple version)
            st.subheader("Keyword Co-occurrence")
            
            # Only attempt if we have real keywords
            if 'keywords' in filtered_df.columns:
                # Get top keywords for analysis
                top_kw_list = top_keywords.index.tolist()[:10]
                
                # Create co-occurrence matrix
                cooccur_matrix = pd.DataFrame(0, index=top_kw_list, columns=top_kw_list)
                
                # Fill the matrix
                for keywords in filtered_df['keywords'].dropna():
                    if keywords.strip():
                        paper_keywords = [kw.strip() for kw in keywords.split(',')]
                        paper_keywords = [kw for kw in paper_keywords if kw in top_kw_list]
                        
                        # Add co-occurrence connections
                        for i, kw1 in enumerate(paper_keywords):
                            for kw2 in paper_keywords[i+1:]:
                                cooccur_matrix.loc[kw1, kw2] += 1
                                cooccur_matrix.loc[kw2, kw1] += 1
                
                # Display as heatmap
                fig = px.imshow(
                    cooccur_matrix,
                    text_auto=True,
                    title='Keyword Co-occurrence Matrix (Top 10 Keywords)',
                    labels=dict(x="Keyword", y="Keyword", color="Co-occurrences"),
                    color_continuous_scale="Greens"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Impact by keyword
            st.subheader("Impact by Keyword")
            
            # Only attempt if we have real keywords
            if 'keywords' in filtered_df.columns:
                # Calculate average citations per keyword
                keyword_impact = {}
                keyword_papers = {}
                
                for _, row in filtered_df.iterrows():
                    if pd.notna(row['keywords']) and row['keywords'].strip():
                        paper_keywords = [kw.strip() for kw in row['keywords'].split(',')]
                        
                        for kw in paper_keywords:
                            if kw not in keyword_impact:
                                keyword_impact[kw] = 0
                                keyword_papers[kw] = 0
                            
                            keyword_impact[kw] += row['citations']
                            keyword_papers[kw] += 1
                
                # Calculate average citations
                keyword_avg_citations = {
                    kw: impact / keyword_papers[kw] 
                    for kw, impact in keyword_impact.items() 
                    if keyword_papers[kw] >= 3  # Only include keywords with minimum papers
                }
                
                # Convert to dataframe
                keyword_impact_df = pd.DataFrame({
                    'Keyword': keyword_avg_citations.keys(),
                    'Avg Citations': keyword_avg_citations.values(),
                    'Paper Count': [keyword_papers[kw] for kw in keyword_avg_citations.keys()]
                })
                
                # Sort and display top impactful keywords
                top_impact_keywords = keyword_impact_df.sort_values('Avg Citations', ascending=False).head(15)
                
                fig = px.scatter(
                    top_impact_keywords,
                    x='Paper Count',
                    y='Avg Citations',
                    text='Keyword',
                    size='Paper Count',
                    color='Avg Citations',
                    title='Most Impactful Keywords (by Average Citations)',
                    labels={'Paper Count': 'Number of Papers', 'Avg Citations': 'Average Citations'}
                )
                
                fig.update_traces(textposition='top center')
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Keyword trends over time
            st.subheader("Keyword Trends Over Time")
            
            # Only attempt if we have real keywords
            if 'keywords' in filtered_df.columns and len(top_keywords) > 0:
                # Get top 5 keywords for trend analysis
                trend_keywords = top_keywords.index.tolist()[:5]
                
                # Let user select keywords to analyze
                selected_keywords = st.multiselect(
                    "Select keywords to analyze trends",
                    options=trend_keywords,
                    default=trend_keywords[:3]
                )
                
                if selected_keywords:
                    # Track keyword usage by year
                    yearly_keyword_data = {}
                    
                    for keyword in selected_keywords:
                        yearly_keyword_data[keyword] = {}
                    
                    for _, row in filtered_df.iterrows():
                        if pd.notna(row['keywords']) and pd.notna(row['publication_date']):
                            year = row['publication_date'].year
                            paper_keywords = [kw.strip() for kw in row['keywords'].split(',')]
                            
                            for keyword in selected_keywords:
                                if keyword in paper_keywords:
                                    if year not in yearly_keyword_data[keyword]:
                                        yearly_keyword_data[keyword][year] = 0
                                    yearly_keyword_data[keyword][year] += 1
                    
                    # Convert to dataframe for plotting
                    trend_data = []
                    for keyword, year_data in yearly_keyword_data.items():
                        for year, count in year_data.items():
                            trend_data.append({
                                'Keyword': keyword,
                                'Year': year,
                                'Count': count
                            })
                    
                    trend_df = pd.DataFrame(trend_data)
                    
                    if not trend_df.empty:
                        fig = px.line(
                            trend_df,
                            x='Year',
                            y='Count',
                            color='Keyword',
                            title='Keyword Trends Over Time',
                            labels={'Year': 'Publication Year', 'Count': 'Number of Papers'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough trend data available for the selected keywords.")
        
        elif analysis_type == "Journal Analysis":
            st.subheader("Journal Impact Analysis")
            
            # Check if journal information is available
            if 'journal' not in filtered_df.columns or filtered_df['journal'].isna().all():
                st.warning("Journal information is not available in the dataset.")
            else:
                # Count papers by journal
                journal_counts = filtered_df['journal'].value_counts()
                
                # Calculate citation metrics by journal
                journal_metrics = filtered_df.groupby('journal').agg({
                    'citations': ['count', 'sum', 'mean', 'median'],
                    'title': 'first'  # Dummy aggregation to avoid errors
                })
                
                # Flatten the multi-index
                journal_metrics.columns = ['paper_count', 'total_citations', 'avg_citations', 'median_citations', 'dummy']
                journal_metrics = journal_metrics.drop(columns=['dummy'])
                journal_metrics = journal_metrics.reset_index()
                
                # Only include journals with at least 2 papers
                journal_metrics = journal_metrics[journal_metrics['paper_count'] >= 2]
                
                # Sort by different metrics
                sort_by = st.radio(
                    "Sort journals by",
                    ["paper_count", "total_citations", "avg_citations"],
                    format_func=lambda x: {
                        "paper_count": "Number of Papers",
                        "total_citations": "Total Citations",
                        "avg_citations": "Average Citations per Paper"
                    }[x],
                    horizontal=True
                )
                
                # Display top journals
                st.subheader("Top Journals")
                
                top_journals = journal_metrics.sort_values(sort_by, ascending=False).head(15)
                
                # Rename columns for display
                display_metrics = top_journals.rename(columns={
                    'journal': 'Journal',
                    'paper_count': 'Papers',
                    'total_citations': 'Total Citations',
                    'avg_citations': 'Avg Citations',
                    'median_citations': 'Median Citations'
                })
                
                st.dataframe(
                    display_metrics,
                    use_container_width=True,
                    column_config={
                        "Papers": st.column_config.NumberColumn(format="%d"),
                        "Total Citations": st.column_config.NumberColumn(format="%d"),
                        "Avg Citations": st.column_config.NumberColumn(format="%.2f"),
                        "Median Citations": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                
                # Visualize top journals
                fig = px.bar(
                    top_journals.head(10),
                    x='journal',
                    y=sort_by,
                    title=f'Top 10 Journals by {sort_by.replace("_", " ").title()}',
                    labels={'journal': 'Journal', sort_by: sort_by.replace("_", " ").title()}
                )
                
                fig.update_layout(xaxis_tickangle=-45)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Journal impact over time
                st.subheader("Journal Impact Over Time")
                
                # Let user select journals to analyze
                top_journal_names = top_journals['journal'].tolist()
                selected_journals = st.multiselect(
                    "Select journals to analyze trends",
                    options=top_journal_names,
                    default=top_journal_names[:3] if len(top_journal_names) >= 3 else top_journal_names
                )
                
                if selected_journals:
                    # Filter data for selected journals
                    journal_data = filtered_df[filtered_df['journal'].isin(selected_journals)]
                    
                    # Group by journal and year
                    journal_data['year'] = journal_data['publication_date'].dt.year
                    journal_yearly = journal_data.groupby(['journal', 'year']).agg({
                        'citations': ['count', 'sum', 'mean']
                    }).reset_index()
                    
                    # Flatten the multi-index
                    journal_yearly.columns = ['journal', 'year', 'paper_count', 'total_citations', 'avg_citations']
                    
                    # Create visualizations
                    trend_type = st.radio(
                        "Trend metric",
                        ["paper_count", "total_citations", "avg_citations"],
                        format_func=lambda x: {
                            "paper_count": "Number of Papers",
                            "total_citations": "Total Citations",
                            "avg_citations": "Average Citations per Paper"
                        }[x],
                        horizontal=True
                    )
                    
                    fig = px.line(
                        journal_yearly,
                        x='year',
                        y=trend_type,
                        color='journal',
                        title=f'Journal Trends: {trend_type.replace("_", " ").title()} Over Time',
                        labels={'year': 'Year', trend_type: trend_type.replace("_", " ").title(), 'journal': 'Journal'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Journal impact comparison
                st.subheader("Journal Impact Comparison")
                
                # Calculate journal impact metrics
                impact_data = []
                
                for _, row in top_journals.iterrows():
                    impact_data.append({
                        'Journal': row['journal'],
                        'Papers': row['paper_count'],
                        'Total Citations': row['total_citations'],
                        'Avg Citations': row['avg_citations'],
                        'Size': row['paper_count']  # For bubble size
                    })
                
                impact_df = pd.DataFrame(impact_data)
                
                # Create bubble chart
                fig = px.scatter(
                    impact_df,
                    x='Papers',
                    y='Avg Citations',
                    size='Size',
                    color='Total Citations',
                    hover_name='Journal',
                    title='Journal Impact Comparison',
                    labels={'Papers': 'Number of Papers', 'Avg Citations': 'Average Citations per Paper'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Temporal Analysis":
            st.subheader("Temporal Impact Analysis")
            
            # Calculate publication and citation metrics by year
            filtered_df['year'] = filtered_df['publication_date'].dt.year
            yearly_metrics = filtered_df.groupby('year').agg({
                'title': 'count',
                'citations': ['sum', 'mean', 'median']
            }).reset_index()
            
            # Flatten the multi-index
            yearly_metrics.columns = ['year', 'publications', 'total_citations', 'avg_citations', 'median_citations']
            
            # Calculate cumulative publications and citations
            yearly_metrics['cum_publications'] = yearly_metrics['publications'].cumsum()
            yearly_metrics['cum_citations'] = yearly_metrics['total_citations'].cumsum()
            
            # Calculate citation rate (citations per year since publication)
            current_year = datetime.now().year
            yearly_metrics['years_since_pub'] = current_year - yearly_metrics['year']
            yearly_metrics['citation_rate'] = yearly_metrics['total_citations'] / yearly_metrics['years_since_pub'].clip(lower=1)
            
            # Publication and citation trends
            st.subheader("Publication and Citation Trends")
            
            trend_options = [
                "publications", "total_citations", "avg_citations", 
                "cum_publications", "cum_citations", "citation_rate"
            ]
            
            selected_trends = st.multiselect(
                "Select trends to display",
                options=trend_options,
                default=["publications", "total_citations"],
                format_func=lambda x: {
                    "publications": "Publications per Year",
                    "total_citations": "Total Citations per Year",
                    "avg_citations": "Average Citations per Paper",
                    "cum_publications": "Cumulative Publications",
                    "cum_citations": "Cumulative Citations",
                    "citation_rate": "Citation Rate (Citations per Year)"
                }[x]
            )
            
            if selected_trends:
                # Create multi-line chart
                fig = go.Figure()
                
                for trend in selected_trends:
                    fig.add_trace(go.Scatter(
                        x=yearly_metrics['year'],
                        y=yearly_metrics[trend],
                        mode='lines+markers',
                        name={
                            "publications": "Publications per Year",
                            "total_citations": "Total Citations per Year",
                            "avg_citations": "Average Citations per Paper",
                            "cum_publications": "Cumulative Publications",
                            "cum_citations": "Cumulative Citations",
                            "citation_rate": "Citation Rate (Citations per Year)"
                        }[trend]
                    ))
                
                fig.update_layout(
                    title="Temporal Trends",
                    xaxis_title="Year",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Citation window analysis
            st.subheader("Citation Window Analysis")
            
            # Calculate citations within specific time windows
            window_sizes = [1, 2, 3, 5]
            window_data = []
            
            for window in window_sizes:
                # Filter to only include publications that have been out for at least the window size
                window_df = filtered_df[filtered_df['year'] <= current_year - window]
                
                if not window_df.empty:
                    avg_citations = window_df['citations'].mean()
                    window_data.append({
                        'Window': f"{window} Year{'s' if window > 1 else ''}",
                        'Avg Citations': avg_citations,
                        'Papers': len(window_df)
                    })
            
            if window_data:
                window_df = pd.DataFrame(window_data)
                
                fig = px.bar(
                    window_df,
                    x='Window',
                    y='Avg Citations',
                    title='Average Citations by Time Window',
                    text='Avg Citations',
                    labels={'Window': 'Citation Window', 'Avg Citations': 'Average Citations per Paper'}
                )
                
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                The Citation Window Analysis shows the average number of citations papers receive within 
                different time windows after publication. This helps understand the citation velocity and 
                compare impact across different time periods.
                """)
            
            # Impact velocity analysis
            st.subheader("Impact Velocity Analysis")
            
            # Create a chart showing how quickly papers are cited
            velocity_df = yearly_metrics.sort_values('year')
            
            if not velocity_df.empty:
                # Calculate normalized metrics for comparison
                scaler = MinMaxScaler()
                velocity_df[['norm_pub', 'norm_cite']] = scaler.fit_transform(
                    velocity_df[['publications', 'total_citations']]
                )
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=velocity_df['year'],
                    y=velocity_df['norm_pub'],
                    mode='lines+markers',
                    name='Normalized Publications'
                ))
                
                fig.add_trace(go.Scatter(
                    x=velocity_df['year'],
                    y=velocity_df['norm_cite'],
                    mode='lines+markers',
                    name='Normalized Citations'
                ))
                
                fig.update_layout(
                    title="Publication vs. Citation Velocity (Normalized)",
                    xaxis_title="Year",
                    yaxis_title="Normalized Value (0-1)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                The Impact Velocity Analysis compares the rate of publications to the rate of citations over time.
                When the citation curve rises more quickly than the publication curve, it indicates accelerating impact.
                """)
