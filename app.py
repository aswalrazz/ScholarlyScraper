import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.api_clients import OpenAlexClient
from utils.data_processing import process_openalex_data, calculate_metrics
from utils.web_scraper import get_website_text_content, enrich_publication_data, find_related_publications

# Page Configuration
st.set_page_config(
    page_title="Scholarly Impact Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_articles' not in st.session_state:
    st.session_state.selected_articles = []
if 'impact_data' not in st.session_state:
    st.session_state.impact_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# App title and description
st.title("Impact Vizor")
st.markdown("""
This visual analytics tool provides insight into the reach and impact of scholarly content. 
Impact Vizor integrates data from the OpenAlex API to help you make evidence-based 
decisions about the impact and resonance of research publications.
""")

# Sidebar for data input and filtering
with st.sidebar:
    st.header("Search Parameters")
    
    search_method = st.radio(
        "Search by:",
        ["Topic/Keyword", "Author", "Journal", "DOI"]
    )
    
    if search_method == "Topic/Keyword":
        search_query = st.text_input("Enter a keyword or topic")
    elif search_method == "Author":
        search_query = st.text_input("Enter author name")
    elif search_method == "Journal":
        search_query = st.text_input("Enter journal name")
    elif search_method == "DOI":
        search_query = st.text_input("Enter DOI")
    
    # Define a min date that goes back to 1900 to allow for historical searches
    min_date = datetime(1900, 1, 1)
    default_start_date = datetime.now() - timedelta(days=365*10)  # Default to 10 years back
    
    date_range = st.date_input(
        "Publication date range",
        value=(default_start_date, datetime.now()),
        min_value=min_date,
        max_value=datetime.now()
    )
    
    # Advanced Filters section
    with st.expander("Advanced Filters"):
        # Publication Type Filter
        publication_types = ["article", "book", "book-chapter", "dissertation", "posted-content", 
                             "proceedings", "reference-entry", "report", "peer-review"]
        selected_types = st.multiselect(
            "Publication Types",
            options=publication_types
        )
        
        # Open Access Filter
        open_access = st.checkbox("Only Open Access")
        
        # Citation Range
        st.subheader("Citation Count Range")
        min_citations, max_citations = st.slider(
            "Citation Range", 
            min_value=0, 
            max_value=10000, 
            value=(0, 10000),
            step=10
        )
        
        # Specific Field/Domain Filter
        fields = ["Biology", "Chemistry", "Computer Science", "Economics", "Engineering", 
                  "Environmental Science", "Mathematics", "Medicine", "Physics", "Psychology", 
                  "Social Sciences"]
        selected_fields = st.multiselect(
            "Research Fields",
            options=fields
        )
        
        # Recent Publication Filter
        recent_only = st.checkbox("Recent Publications Only (Last 2 Years)")
        
        # Language Filter
        languages = ["English", "Chinese", "Spanish", "German", "French", "Japanese"]
        selected_languages = st.multiselect(
            "Languages",
            options=languages
        )
    
    # Show me active filters summary
    if (selected_types or open_access or min_citations > 0 or max_citations < 10000 or 
        selected_fields or recent_only or selected_languages):
        
        st.subheader("Active Filters")
        active_filters = []
        
        if selected_types:
            active_filters.append(f"Types: {', '.join(selected_types)}")
        if open_access:
            active_filters.append("Open Access Only")
        if min_citations > 0 or max_citations < 10000:
            active_filters.append(f"Citations: {min_citations} - {max_citations}")
        if selected_fields:
            active_filters.append(f"Fields: {', '.join(selected_fields)}")
        if recent_only:
            active_filters.append("Recent Publications Only")
        if selected_languages:
            active_filters.append(f"Languages: {', '.join(selected_languages)}")
        
        for filter_text in active_filters:
            st.markdown(f"* {filter_text}")
    
    search_button = st.button("Search", use_container_width=True)

# If search button is clicked
if search_button and search_query:
    with st.spinner("Fetching data from scholarly databases..."):
        # Initialize API client
        openalex_client = OpenAlexClient()
        
        start_date = date_range[0].strftime("%Y-%m-%d")
        end_date = date_range[1].strftime("%Y-%m-%d") if len(date_range) > 1 else datetime.now().strftime("%Y-%m-%d")
        
        # Handle the recent publication filter
        if 'recent_only' in locals() and recent_only:
            current_date = datetime.now()
            two_years_ago = (current_date - timedelta(days=365*2)).strftime("%Y-%m-%d")
            start_date = two_years_ago
        
        # Set additional filter parameters based on search method and advanced filters
        filters = {
            "publication_date": f"{start_date}:{end_date}"
        }
        
        # Add publication type filter
        if 'selected_types' in locals() and selected_types:
            type_filters = []
            for pub_type in selected_types:
                type_filters.append(pub_type)
            
            if type_filters:
                filters["type"] = "|".join(type_filters)  # Use OR operator for types
        
        # Add open access filter
        if 'open_access' in locals() and open_access:
            filters["is_oa"] = "true"
        
        # Add field/domain filter
        if 'selected_fields' in locals() and selected_fields:
            field_filters = []
            for field in selected_fields:
                field_filters.append(field)
            
            if field_filters:
                filters["concepts.display_name"] = "|".join(field_filters)  # Use OR operator for fields
        
        # Add language filter
        if 'selected_languages' in locals() and selected_languages:
            language_filters = []
            for language in selected_languages:
                language_filters.append(language)
            
            if language_filters:
                filters["language"] = "|".join(language_filters)  # Use OR operator for languages
        
        st.write(f"Using filters: {filters}")  # Debug: show filters being used
        
        # Perform search based on selected method
        if search_method == "DOI":
            results = openalex_client.get_work_by_doi(search_query)
        elif search_method == "Author":
            results = openalex_client.search_works(
                query=search_query,
                filter_field="author.display_name",
                filter_value=search_query,
                additional_filters=filters
            )
        elif search_method == "Journal":
            results = openalex_client.search_works(
                query=search_query,
                filter_field="host_venue.display_name",
                filter_value=search_query,
                additional_filters=filters
            )
        else:  # Topic/Keyword
            results = openalex_client.search_works(
                query=search_query,
                filter_field="publication_date",
                filter_value=f"{start_date}:{end_date}",
                additional_filters={k: v for k, v in filters.items() if k != "publication_date"}
            )
        
        # Process results
        processed_results = process_openalex_data(results)
        
        if isinstance(processed_results, pd.DataFrame) and not processed_results.empty:
            # Apply client-side filtering for citation range
            if 'min_citations' in locals() and 'max_citations' in locals():
                if min_citations > 0 or max_citations < 10000:
                    processed_results = processed_results[
                        (processed_results['citations'] >= min_citations) & 
                        (processed_results['citations'] <= max_citations)
                    ]
            
            if not processed_results.empty:
                st.session_state.search_results = processed_results
                st.session_state.search_performed = True
                
                # Calculate impact metrics
                st.session_state.metrics = calculate_metrics(processed_results)
            else:
                st.error("No results match your citation filters. Please adjust the citation range.")
                st.session_state.search_performed = False
        else:
            st.error("No results found for your search criteria. Please try different keywords or filters.")
            st.session_state.search_performed = False

# Display search results
if st.session_state.search_performed and st.session_state.search_results is not None:
    st.subheader("Search Results")
    
    # Display overview metrics
    if st.session_state.metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Publications", st.session_state.metrics["total_publications"])
        col2.metric("Total Citations", st.session_state.metrics["total_citations"])
        col3.metric("Avg. Citations per Article", f"{st.session_state.metrics['avg_citations']:.2f}")
        col4.metric("h-index", st.session_state.metrics["h_index"])
    
    # Define columns to display in the main table
    display_columns = [
        'title', 'authors', 'year', 'publication_date', 'source', 
        'institutions', 'citations', 'cited_by', 'related_count', 
        'fwci', 'citation_percentile', 'h_index_contribution',
        'type', 'topic', 'subfield', 'field', 'domain', 'open_access_status', 'doi'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in st.session_state.search_results.columns]
    
    # Prepare DOI links by adding the https://doi.org/ prefix if not present
    if 'doi' in st.session_state.search_results.columns:
        st.session_state.search_results['doi_url'] = st.session_state.search_results['doi'].apply(
            lambda x: f"https://doi.org/{x}" if x and not str(x).startswith('http') else x
        )
        available_columns = [col for col in available_columns if col != 'doi'] + ['doi_url']
    
    # Configure column display with appropriate formatting
    column_config = {
        "title": st.column_config.TextColumn("Title"),
        "authors": st.column_config.TextColumn("Authors"),
        "year": st.column_config.NumberColumn("Year", format="%d"),
        "publication_date": st.column_config.DateColumn("Publication Date"),
        "source": st.column_config.TextColumn("Source/Journal"),
        "institutions": st.column_config.TextColumn("Institutions"),
        "citations": st.column_config.NumberColumn("Citations", format="%d"),
        "cited_by": st.column_config.NumberColumn("Cited By", format="%d"),
        "related_count": st.column_config.NumberColumn("Related Works", format="%d"),
        "fwci": st.column_config.NumberColumn("FWCI", format="%.3f"),
        "citation_percentile": st.column_config.NumberColumn("Citation Percentile", format="%.2f"),
        "h_index_contribution": st.column_config.NumberColumn("H-index", format="%d"),
        "type": st.column_config.TextColumn("Type"),
        "topic": st.column_config.TextColumn("Topic"),
        "subfield": st.column_config.TextColumn("Subfield"),
        "field": st.column_config.TextColumn("Field"),
        "domain": st.column_config.TextColumn("Domain"),
        "open_access_status": st.column_config.TextColumn("Open Access"),
        "doi_url": st.column_config.LinkColumn("DOI", display_text="View", width="small"),
    }
    
    # Advanced settings UI
    with st.expander("Table Display Settings"):
        # Allow user to select which columns to display
        selected_columns = st.multiselect(
            "Select columns to display",
            options=available_columns,
            default=['title', 'authors', 'year', 'institutions', 'citations', 'doi_url']
        )
        
        if selected_columns:
            display_columns = selected_columns
        else:
            # Use default subset if nothing selected
            default_columns = ['title', 'authors', 'year', 'institutions', 'citations', 'doi_url']
            display_columns = [col for col in default_columns if col in available_columns]
        
        # Option to sort the table
        sort_options = [col for col in ['citations', 'year', 'publication_date', 'title', 'fwci', 'citation_percentile'] 
                        if col in st.session_state.search_results.columns]
        
        if sort_options:
            sort_by = st.selectbox(
                "Sort results by",
                options=sort_options,
                index=0 if 'citations' in sort_options else 0
            )
            
            sort_order = st.radio("Sort order", options=["Descending", "Ascending"], horizontal=True)
            ascending = sort_order == "Ascending"
            
            # Sort the dataframe based on user selection
            st.session_state.search_results = st.session_state.search_results.sort_values(
                by=sort_by, 
                ascending=ascending
            )
    
    # Display the table with the configured columns and filtering to available columns
    valid_display_columns = [col for col in display_columns if col in st.session_state.search_results.columns]
    
    st.dataframe(
        st.session_state.search_results[valid_display_columns],
        use_container_width=True,
        column_config={col: config for col, config in column_config.items() if col in valid_display_columns},
        height=400
    )
    
    # Provide option to download the full results
    csv = st.session_state.search_results.to_csv(index=False)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download results as CSV",
            csv,
            "impact_vizor_results.csv",
            "text/csv",
            key='download-csv'
        )
    
    # Add option to enrich data with web scraping (for abstracts, etc.)
    with col2:
        if st.button("Enrich Data with Web Scraping"):
            with st.spinner("Enriching publication data by fetching additional information..."):
                # Take a sample to avoid too many requests for large result sets
                max_items = 10 if len(st.session_state.search_results) > 10 else len(st.session_state.search_results)
                st.info(f"Processing {max_items} publications to enhance with additional data...")
                
                # Use the enrich_publication_data function from the web scraper
                enriched_df = enrich_publication_data(st.session_state.search_results, max_items=max_items)
                
                # Update the session state with the enriched data
                st.session_state.search_results = enriched_df
                
                # Show success message
                st.success(f"Successfully enhanced {max_items} publications with additional data where available.")
                
                # Provide the enhanced download option
                enriched_csv = enriched_df.to_csv(index=False)
                st.download_button(
                    "Download Enhanced Results as CSV",
                    enriched_csv,
                    "impact_vizor_enriched_results.csv",
                    "text/csv",
                    key='download-enriched-csv'
                )
    
    # Detailed paper view section
    st.subheader("Detailed Publication Information")
    paper_selector = st.selectbox(
        "Select a publication to view detailed information", 
        options=st.session_state.search_results['title'].tolist(),
        format_func=lambda x: x[:100] + "..." if len(x) > 100 else x
    )
    
    if paper_selector:
        selected_paper = st.session_state.search_results[st.session_state.search_results['title'] == paper_selector].iloc[0]
        
        with st.container():
            st.markdown(f"## {selected_paper['title']}")
            
            # Paper metadata in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Publication Info")
                st.markdown(f"**Year:** {selected_paper.get('year', '')}")
                st.markdown(f"**Type:** {selected_paper.get('type', '')}")
                st.markdown(f"**Source:** {selected_paper.get('source', selected_paper.get('journal', ''))}")
                st.markdown(f"**Open Access:** {selected_paper.get('open_access_status', '')}")
                if 'doi' in selected_paper and selected_paper['doi']:
                    st.markdown(f"**DOI:** [View Paper](https://doi.org/{selected_paper['doi']})")
            
            with col2:
                st.markdown("### Author Information")
                st.markdown(f"**Authors:** {selected_paper.get('authors', '')}")
                st.markdown(f"**Institutions:** {selected_paper.get('institutions', '')}")
            
            with col3:
                st.markdown("### Impact Metrics")
                st.markdown(f"**Citations:** {int(selected_paper.get('citations', 0))}")
                st.markdown(f"**Cited by:** {int(selected_paper.get('cited_by', 0))}")
                st.markdown(f"**Related papers:** {int(selected_paper.get('related_count', 0))}")
                st.markdown(f"**FWCI:** {selected_paper.get('fwci', 0):.3f}")
                st.markdown(f"**Citation percentile:** {selected_paper.get('citation_percentile', 0):.2f}")
                st.markdown(f"**H-index contribution:** {int(selected_paper.get('h_index_contribution', 0))}")
            
            # Subject classification
            st.markdown("### Subject Classification")
            subj_cols = st.columns(4)
            with subj_cols[0]:
                st.markdown(f"**Topic:** {selected_paper.get('topic', '')}")
            with subj_cols[1]:
                st.markdown(f"**Subfield:** {selected_paper.get('subfield', '')}")
            with subj_cols[2]:
                st.markdown(f"**Field:** {selected_paper.get('field', '')}")
            with subj_cols[3]:
                st.markdown(f"**Domain:** {selected_paper.get('domain', '')}")
            
            # Abstract section
            st.markdown("### Abstract")
            
            # Check if abstract is available
            has_abstract = 'abstract' in selected_paper and selected_paper['abstract']
            
            if has_abstract:
                st.markdown(selected_paper['abstract'])
            else:
                # If no abstract is available, offer to fetch it from the web
                if 'doi' in selected_paper and selected_paper['doi']:
                    doi_url = f"https://doi.org/{selected_paper['doi']}" if not str(selected_paper['doi']).startswith('http') else selected_paper['doi']
                    
                    if st.button("Fetch Abstract from Web", key="fetch_abstract"):
                        with st.spinner("Fetching abstract from publication source..."):
                            # Try to scrape content from the DOI URL
                            st.info(f"Attempting to retrieve content from {doi_url}")
                            content = get_website_text_content(doi_url)
                            
                            if content:
                                # Extract first 1000 characters as the abstract (simple approach)
                                # In a real-world scenario, we might use NLP to extract the proper abstract
                                abstract_preview = content[:1000] + "..." if len(content) > 1000 else content
                                st.markdown(abstract_preview)
                                
                                # Store the full text for further analysis
                                if 'full_text' not in st.session_state:
                                    st.session_state.full_text = {}
                                st.session_state.full_text[selected_paper['doi']] = content
                                
                                # Option to view full text
                                if st.button("View Full Text", key="view_full_text"):
                                    st.text_area("Full Publication Text", content, height=400)
                            else:
                                st.warning("Could not extract content from the publication source. The publisher may restrict access.")
                    else:
                        st.info("No abstract available. Click the button above to attempt retrieving content from the publication source.")
                else:
                    st.info("No abstract available and no DOI link to fetch content from.")
            
            # Add related publications section
            if 'doi' in selected_paper and selected_paper['doi']:
                st.markdown("### Related Publications")
                doi_url = f"https://doi.org/{selected_paper['doi']}" if not str(selected_paper['doi']).startswith('http') else selected_paper['doi']
                
                # Check if we've already fetched related publications for this DOI
                if 'related_pubs' not in st.session_state:
                    st.session_state.related_pubs = {}
                
                if selected_paper['doi'] in st.session_state.related_pubs:
                    # Show cached related publications
                    related_links = st.session_state.related_pubs[selected_paper['doi']]
                    if related_links:
                        for i, link in enumerate(related_links):
                            st.markdown(f"{i+1}. [{link}]({link})")
                    else:
                        st.info("No related publications found.")
                else:
                    # Show button to fetch related publications
                    if st.button("Find Related Publications", key="find_related"):
                        with st.spinner("Searching for related publications..."):
                            # Use the web scraper to find related publications
                            related_links = find_related_publications(doi_url)
                            
                            # Cache the results
                            st.session_state.related_pubs[selected_paper['doi']] = related_links
                            
                            if related_links:
                                for i, link in enumerate(related_links):
                                    st.markdown(f"{i+1}. [{link}]({link})")
                            else:
                                st.info("No related publications found. This may be due to access restrictions on the publication page.")
    
    # Add interactive visualizations
    if not st.session_state.search_results.empty:
        st.subheader("Impact Visualization")
        
        # Visualization customization options
        with st.expander("Visualization Settings"):
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Color scheme selection
                color_schemes = {
                    "Default": ["#3366CC", "#FF9900", "#109618", "#DC3912", "#990099"],
                    "Viridis": px.colors.sequential.Viridis,
                    "Plasma": px.colors.sequential.Plasma,
                    "Blues": px.colors.sequential.Blues,
                    "Reds": px.colors.sequential.Reds,
                    "Greens": px.colors.sequential.Greens,
                    "Spectral": px.colors.diverging.Spectral
                }
                
                selected_color_scheme = st.selectbox(
                    "Color Scheme",
                    options=list(color_schemes.keys()),
                    index=0
                )
                
                # Line/marker style
                line_styles = ["solid", "dot", "dash", "longdash", "dashdot"]
                selected_line_style = st.selectbox(
                    "Line Style",
                    options=line_styles,
                    index=0
                )
                
                # Show gridlines
                show_grid = st.checkbox("Show Grid Lines", value=True)
                
            with viz_col2:
                # Chart type
                chart_types = ["Line", "Bar", "Area"]
                selected_chart_type = st.selectbox(
                    "Chart Type",
                    options=chart_types,
                    index=0
                )
                
                # Smoothing option
                smoothing_enabled = st.checkbox("Enable Smoothing", value=True)
                smoothing_line_shape = "spline" if smoothing_enabled else "linear"
                
                # Animation option
                animation_enabled = st.checkbox("Enable Animation", value=False)
        
        # Set up tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Publications Timeline", "Citation Analysis", "Author Metrics"])
        
        with tab1:
            # Publications over time with trend
            df_grouped = st.session_state.search_results.copy()
            df_grouped['year'] = pd.to_datetime(df_grouped['publication_date']).dt.year
            
            # Aggregate by year
            pub_by_year = df_grouped.groupby('year').size().reset_index(name='count')
            
            # Calculate cumulative publications
            pub_by_year['cumulative'] = pub_by_year['count'].cumsum()
            
            # Year on year growth rate
            pub_by_year['growth_rate'] = pub_by_year['count'].pct_change() * 100
            
            # Apply visualization customization options
            # Get selected color scheme
            color_sequence = color_schemes.get(selected_color_scheme, color_schemes["Default"])
            
            # Create visualization based on selected chart type
            if selected_chart_type == "Bar":
                fig1 = px.bar(
                    pub_by_year, 
                    x='year', 
                    y=['count', 'cumulative'],
                    title='Publication Trends Over Time',
                    labels={
                        'count': 'Annual Publications',
                        'cumulative': 'Cumulative Publications',
                        'year': 'Year'
                    },
                    color_discrete_sequence=color_sequence,
                    barmode='group'
                )
            elif selected_chart_type == "Area":
                fig1 = px.area(
                    pub_by_year, 
                    x='year', 
                    y=['count', 'cumulative'],
                    title='Publication Trends Over Time',
                    labels={
                        'count': 'Annual Publications',
                        'cumulative': 'Cumulative Publications',
                        'year': 'Year'
                    },
                    color_discrete_sequence=color_sequence
                )
            else:  # Line chart (default)
                fig1 = px.line(
                    pub_by_year, 
                    x='year', 
                    y=['count', 'cumulative'],
                    title='Publication Trends Over Time',
                    labels={
                        'count': 'Annual Publications',
                        'cumulative': 'Cumulative Publications',
                        'year': 'Year'
                    },
                    line_shape=smoothing_line_shape,  # Apply smoothing setting
                    markers=True,  # Show markers
                    color_discrete_sequence=color_sequence
                )
            
            # Apply line style if it's a line chart
            if selected_chart_type == "Line":
                for trace in fig1.data:
                    trace.update(line=dict(dash=selected_line_style))
            
            # Improve layout
            fig1.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray' if show_grid else None,
                    showgrid=show_grid,
                    title=dict(font=dict(size=14)),
                ),
                yaxis=dict(
                    gridcolor='lightgray' if show_grid else None,
                    showgrid=show_grid,
                    title=dict(font=dict(size=14)),
                )
            )
            
            # Add animation if enabled
            if animation_enabled:
                fig1.update_layout(
                    updatemenus=[{
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [{
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        }]
                    }]
                )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show growth rate bar chart if we have enough data
            if len(pub_by_year) > 2:
                growth_data = pub_by_year.dropna(subset=['growth_rate'])
                
                if not growth_data.empty:
                    fig_growth = px.bar(
                        growth_data,
                        x='year',
                        y='growth_rate',
                        title='Year-on-Year Publication Growth Rate (%)',
                        color='growth_rate',
                        color_continuous_scale=px.colors.diverging.RdBu,
                        color_continuous_midpoint=0,
                        text='growth_rate'
                    )
                    
                    fig_growth.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside'
                    )
                    
                    fig_growth.update_layout(
                        plot_bgcolor='white',
                        xaxis=dict(gridcolor='lightgray'),
                        yaxis=dict(gridcolor='lightgray')
                    )
                    
                    st.plotly_chart(fig_growth, use_container_width=True)
        
        with tab2:
            if 'citations' in st.session_state.search_results.columns:
                col1, col2 = st.columns(2)
                
                # Calculate citation metrics
                total_citations = int(st.session_state.search_results['citations'].sum())
                avg_citations = st.session_state.search_results['citations'].mean()
                max_citations = st.session_state.search_results['citations'].max()
                citation_median = st.session_state.search_results['citations'].median()
                
                with col1:
                    st.metric("Total Citations", f"{total_citations:,}")
                    st.metric("Maximum Citations", f"{int(max_citations):,}")
                
                with col2:
                    st.metric("Average Citations", f"{avg_citations:.2f}")
                    st.metric("Median Citations", f"{citation_median:.1f}")
                
                # Enhanced citation distribution with log scale option
                use_log = st.checkbox("Use logarithmic scale", value=True)
                
                # Calculate percentiles for outlier detection
                q75 = st.session_state.search_results['citations'].quantile(0.75)
                q25 = st.session_state.search_results['citations'].quantile(0.25)
                iqr = q75 - q25
                outlier_cutoff = q75 + 1.5 * iqr
                
                # Identify highly cited papers (outliers)
                highly_cited = st.session_state.search_results[
                    st.session_state.search_results['citations'] > outlier_cutoff
                ]
                
                # Apply visualization customization options
                # Get selected color scheme for consistency across visualizations
                color_sequence = color_schemes.get(selected_color_scheme, color_schemes["Default"])
                
                fig2 = px.histogram(
                    st.session_state.search_results,
                    x='citations',
                    title='Citation Distribution',
                    labels={'citations': 'Number of Citations', 'count': 'Number of Articles'},
                    nbins=30,
                    opacity=0.8,
                    color_discrete_sequence=[color_sequence[0]]  # Use first color from selected scheme
                )
                
                if use_log:
                    fig2.update_layout(xaxis_type="log")
                
                fig2.update_layout(
                    bargap=0.1,
                    plot_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray' if show_grid else None,
                        showgrid=show_grid,
                        title=dict(font=dict(size=14))
                    ),
                    yaxis=dict(
                        gridcolor='lightgray' if show_grid else None,
                        showgrid=show_grid,
                        title=dict(font=dict(size=14))
                    )
                )
                
                # Add animation if enabled
                if animation_enabled:
                    fig2.update_layout(
                        updatemenus=[{
                            "type": "buttons",
                            "showactive": False,
                            "buttons": [{
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                            }]
                        }]
                    )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show top cited papers
                st.subheader("Most Cited Publications")
                top_cited = st.session_state.search_results.sort_values('citations', ascending=False).head(5)
                
                # Create a simple horizontal bar chart for top cited papers
                # Choose appropriate color scale based on selected scheme
                if selected_color_scheme == "Viridis":
                    color_scale = px.colors.sequential.Viridis
                elif selected_color_scheme == "Plasma":
                    color_scale = px.colors.sequential.Plasma
                elif selected_color_scheme == "Blues":
                    color_scale = px.colors.sequential.Blues
                elif selected_color_scheme == "Reds":
                    color_scale = px.colors.sequential.Reds
                elif selected_color_scheme == "Greens":
                    color_scale = px.colors.sequential.Greens
                elif selected_color_scheme == "Spectral":
                    color_scale = px.colors.diverging.Spectral
                else:
                    color_scale = px.colors.sequential.Viridis
                
                fig_top = px.bar(
                    top_cited,
                    y='title',
                    x='citations',
                    orientation='h',
                    title='Top 5 Most Cited Publications',
                    text='citations',
                    color='citations',
                    color_continuous_scale=color_scale
                )
                
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(
                    plot_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray' if show_grid else None,
                        showgrid=show_grid,
                        title=dict(font=dict(size=14))
                    ),
                    yaxis=dict(
                        categoryorder='total ascending',
                        gridcolor='lightgray' if show_grid else None,
                        showgrid=show_grid
                    )
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.warning("Citation data is not available.")
        
        with tab3:
            # Extract top authors
            if 'authors' in st.session_state.search_results.columns:
                # Process authors data
                all_authors = []
                for authors_str in st.session_state.search_results['authors']:
                    if pd.notna(authors_str) and authors_str:
                        author_list = [a.strip() for a in authors_str.split(',')]
                        all_authors.extend(author_list)
                
                if all_authors:
                    # Count frequency
                    author_counts = pd.Series(all_authors).value_counts()
                    top_authors = author_counts.head(10)
                    
                    # Choose appropriate color scale based on selected scheme
                    if selected_color_scheme == "Viridis":
                        color_scale = px.colors.sequential.Viridis
                    elif selected_color_scheme == "Plasma":
                        color_scale = px.colors.sequential.Plasma
                    elif selected_color_scheme == "Blues":
                        color_scale = px.colors.sequential.Blues
                    elif selected_color_scheme == "Reds":
                        color_scale = px.colors.sequential.Reds
                    elif selected_color_scheme == "Greens":
                        color_scale = px.colors.sequential.Greens
                    elif selected_color_scheme == "Spectral":
                        color_scale = px.colors.diverging.Spectral
                    else:
                        color_scale = px.colors.sequential.Viridis
                    
                    # Use the same color scale as previous charts for consistency
                    fig_authors = px.bar(
                        x=top_authors.index,
                        y=top_authors.values,
                        title='Top 10 Contributing Authors',
                        labels={'x': 'Author', 'y': 'Number of Publications'},
                        color=top_authors.values,
                        color_continuous_scale=color_scale,  # Use the same color scale
                        text=top_authors.values
                    )
                    
                    fig_authors.update_traces(textposition='outside')
                    fig_authors.update_layout(
                        plot_bgcolor='white',
                        xaxis=dict(
                            gridcolor='lightgray' if show_grid else None,
                            showgrid=show_grid,
                            tickangle=45
                        ),
                        yaxis=dict(
                            gridcolor='lightgray' if show_grid else None,
                            showgrid=show_grid
                        )
                    )
                    
                    # Add animation if enabled
                    if animation_enabled:
                        fig_authors.update_layout(
                            updatemenus=[{
                                "type": "buttons",
                                "showactive": False,
                                "buttons": [{
                                    "label": "Play",
                                    "method": "animate",
                                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                                }]
                            }]
                        )
                    
                    st.plotly_chart(fig_authors, use_container_width=True)
                else:
                    st.info("No author information available in the dataset.")
            else:
                st.info("Author information is not available in the dataset.")
else:
    # Display welcome message and instructions when no search has been performed
    st.info("""
    ### Welcome to Impact Vizor!
    
    To get started:
    1. Use the sidebar to set your search parameters
    2. Click the "Search" button to retrieve scholarly impact data
    3. Explore the various visualizations and analytics tools available in the pages
    
    Navigate between different analytics views using the page selector in the sidebar.
    """)

# Footer with info about data sources
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
    Data sourced from OpenAlex and Crossref APIs. Last updated: {}
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")),
    unsafe_allow_html=True
)
