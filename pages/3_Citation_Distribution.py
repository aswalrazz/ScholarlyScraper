import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(page_title="Citation Distribution", page_icon="📊", layout="wide")

# Page title
st.title("Citation Distribution Surveyor")
st.markdown("Analyze the distribution of citations across publications")

# Check if search has been performed and data exists
if 'search_performed' not in st.session_state or not st.session_state.search_performed:
    st.info("Please perform a search on the main page first to analyze citation distributions.")
elif 'search_results' not in st.session_state or st.session_state.search_results is None or st.session_state.search_results.empty:
    st.warning("No data available for analysis. Please search for scholarly content on the main page.")
else:
    # Get data from session state
    df = st.session_state.search_results
    
    # Sidebar for filtering options
    with st.sidebar:
        st.header("Analysis Options")
        
        # Select time period for analysis
        time_options = ["All Time", "Last 5 Years", "Last 10 Years", "Custom Range"]
        time_period = st.radio("Time Period", time_options)
        
        if time_period == "Custom Range":
            min_year = int(df['publication_date'].dt.year.min())
            max_year = int(datetime.now().year)
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        # Select grouping variable
        group_options = ["None", "Journal", "Year", "Article Type"]
        grouping_var = st.selectbox("Group By", group_options)
        
        # Select visualization type
        viz_options = ["Histogram", "Box Plot", "Violin Plot", "ECDF", "Percentile Chart"]
        viz_type = st.selectbox("Visualization Type", viz_options)
        
        # Additional options for histogram
        if viz_type == "Histogram":
            bin_count = st.slider("Number of Bins", 5, 100, 20)
            log_scale = st.checkbox("Use Log Scale for X-axis", value=True)
        
    # Filter data based on selected time period
    filtered_df = df.copy()
    
    if time_period == "Last 5 Years":
        cutoff_year = datetime.now().year - 5
        filtered_df = filtered_df[filtered_df['publication_date'].dt.year >= cutoff_year]
    elif time_period == "Last 10 Years":
        cutoff_year = datetime.now().year - 10
        filtered_df = filtered_df[filtered_df['publication_date'].dt.year >= cutoff_year]
    elif time_period == "Custom Range":
        filtered_df = filtered_df[
            (filtered_df['publication_date'].dt.year >= year_range[0]) &
            (filtered_df['publication_date'].dt.year <= year_range[1])
        ]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No data available for the selected time period. Please adjust your filters.")
    else:
        # Prepare data based on grouping variable
        if grouping_var == "None":
            # No grouping, use the filtered data as is
            data_for_viz = filtered_df
            group_col = None
        elif grouping_var == "Journal":
            # Group by journal
            if 'journal' in filtered_df.columns:
                # Keep only top journals by publication count for readability
                journal_counts = filtered_df['journal'].value_counts()
                top_journals = journal_counts[journal_counts >= 3].index.tolist()
                
                if top_journals:
                    data_for_viz = filtered_df[filtered_df['journal'].isin(top_journals)]
                    group_col = 'journal'
                else:
                    data_for_viz = filtered_df
                    group_col = None
                    st.warning("Not enough data to group by journal. Showing ungrouped visualization.")
            else:
                data_for_viz = filtered_df
                group_col = None
                st.warning("Journal information not available. Showing ungrouped visualization.")
        elif grouping_var == "Year":
            # Group by publication year
            data_for_viz = filtered_df
            data_for_viz['year'] = data_for_viz['publication_date'].dt.year
            group_col = 'year'
        elif grouping_var == "Article Type":
            # Group by article type if available
            if 'type' in filtered_df.columns:
                data_for_viz = filtered_df
                group_col = 'type'
            else:
                data_for_viz = filtered_df
                group_col = None
                st.warning("Article type information not available. Showing ungrouped visualization.")
        
        # Create visualizations based on selected type
        st.subheader(f"Citation Distribution {f'by {grouping_var}' if group_col else ''}")
        
        if viz_type == "Histogram":
            if group_col:
                fig = px.histogram(
                    data_for_viz,
                    x='citations',
                    color=group_col,
                    nbins=bin_count,
                    title=f'Citation Distribution by {grouping_var}',
                    labels={'citations': 'Number of Citations', 'count': 'Frequency'},
                    log_x=log_scale,
                    opacity=0.7,
                    barmode='overlay'
                )
            else:
                fig = px.histogram(
                    data_for_viz,
                    x='citations',
                    nbins=bin_count,
                    title='Citation Distribution',
                    labels={'citations': 'Number of Citations', 'count': 'Frequency'},
                    log_x=log_scale
                )
            
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Box Plot":
            if group_col:
                fig = px.box(
                    data_for_viz,
                    x=group_col,
                    y='citations',
                    title=f'Citation Distribution by {grouping_var}',
                    labels={group_col: grouping_var, 'citations': 'Number of Citations'}
                )
            else:
                # For no grouping, create a single box plot
                fig = px.box(
                    data_for_viz,
                    y='citations',
                    title='Citation Distribution',
                    labels={'citations': 'Number of Citations'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Violin Plot":
            if group_col:
                fig = px.violin(
                    data_for_viz,
                    x=group_col,
                    y='citations',
                    box=True,
                    points="all",
                    title=f'Citation Distribution by {grouping_var}',
                    labels={group_col: grouping_var, 'citations': 'Number of Citations'}
                )
            else:
                # For no grouping, create a single violin plot
                fig = px.violin(
                    data_for_viz,
                    y='citations',
                    box=True,
                    points="all",
                    title='Citation Distribution',
                    labels={'citations': 'Number of Citations'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "ECDF":
            # Empirical Cumulative Distribution Function
            if group_col:
                fig = px.ecdf(
                    data_for_viz,
                    x='citations',
                    color=group_col,
                    title=f'Cumulative Citation Distribution by {grouping_var}',
                    labels={'citations': 'Number of Citations', 'ecdf': 'Cumulative Proportion'}
                )
            else:
                fig = px.ecdf(
                    data_for_viz,
                    x='citations',
                    title='Cumulative Citation Distribution',
                    labels={'citations': 'Number of Citations', 'ecdf': 'Cumulative Proportion'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Percentile Chart":
            # Calculate percentiles
            percentiles = range(0, 101, 5)
            
            if group_col:
                # Calculate percentiles for each group
                percentile_data = []
                
                for group_name, group_df in data_for_viz.groupby(group_col):
                    if len(group_df) > 5:  # Only include groups with enough data
                        group_percentiles = np.percentile(group_df['citations'], percentiles)
                        for p, value in zip(percentiles, group_percentiles):
                            percentile_data.append({
                                'Percentile': p,
                                'Citations': value,
                                grouping_var: group_name
                            })
                
                if percentile_data:
                    percentile_df = pd.DataFrame(percentile_data)
                    
                    fig = px.line(
                        percentile_df,
                        x='Percentile',
                        y='Citations',
                        color=grouping_var,
                        title=f'Citation Percentiles by {grouping_var}',
                        labels={'Percentile': 'Percentile', 'Citations': 'Number of Citations'}
                    )
                else:
                    st.warning(f"Not enough data in groups to calculate percentiles for {grouping_var}.")
                    # Fall back to overall percentiles
                    overall_percentiles = np.percentile(data_for_viz['citations'], percentiles)
                    percentile_df = pd.DataFrame({
                        'Percentile': percentiles,
                        'Citations': overall_percentiles
                    })
                    
                    fig = px.line(
                        percentile_df,
                        x='Percentile',
                        y='Citations',
                        title='Citation Percentiles (Overall)',
                        labels={'Percentile': 'Percentile', 'Citations': 'Number of Citations'}
                    )
            else:
                # Calculate overall percentiles
                overall_percentiles = np.percentile(data_for_viz['citations'], percentiles)
                percentile_df = pd.DataFrame({
                    'Percentile': percentiles,
                    'Citations': overall_percentiles
                })
                
                fig = px.line(
                    percentile_df,
                    x='Percentile',
                    y='Citations',
                    title='Citation Percentiles',
                    labels={'Percentile': 'Percentile', 'Citations': 'Number of Citations'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Citation Summary Statistics")
        
        if group_col:
            # Calculate statistics by group
            stats_df = data_for_viz.groupby(group_col)['citations'].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max')
            ]).reset_index()
            
            # Round statistics
            stats_df['mean'] = stats_df['mean'].round(2)
            stats_df['median'] = stats_df['median'].round(2)
            stats_df['std'] = stats_df['std'].round(2)
            
            # Rename columns
            stats_df.columns = [grouping_var, 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
            
            st.dataframe(stats_df, use_container_width=True)
        else:
            # Calculate overall statistics
            count = len(data_for_viz)
            mean = data_for_viz['citations'].mean()
            median = data_for_viz['citations'].median()
            std = data_for_viz['citations'].std()
            min_val = data_for_viz['citations'].min()
            max_val = data_for_viz['citations'].max()
            
            # Create columns for display
            col1, col2, col3 = st.columns(3)
            col1.metric("Count", count)
            col1.metric("Mean", f"{mean:.2f}")
            col2.metric("Median", f"{median:.2f}")
            col2.metric("Std Dev", f"{std:.2f}")
            col3.metric("Min", min_val)
            col3.metric("Max", max_val)
        
        # Show top and bottom papers by citation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Cited Papers")
            top_papers = data_for_viz.sort_values('citations', ascending=False).head(5)
            st.dataframe(
                top_papers[['title', 'authors', 'publication_date', 'citations', 'doi']],
                use_container_width=True,
                column_config={
                    "citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "doi": st.column_config.LinkColumn("DOI", display_text="View"),
                }
            )
        
        with col2:
            st.subheader("Least Cited Papers")
            bottom_papers = data_for_viz.sort_values('citations').head(5)
            st.dataframe(
                bottom_papers[['title', 'authors', 'publication_date', 'citations', 'doi']],
                use_container_width=True,
                column_config={
                    "citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "doi": st.column_config.LinkColumn("DOI", display_text="View"),
                }
            )
