import requests
import pandas as pd
import time
from datetime import datetime
import os
import urllib.parse

class OpenAlexClient:
    """Client for interacting with the OpenAlex API"""
    
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        # Set user email for polite pool
        self.email = os.getenv("USER_EMAIL", "user@example.com")
        self.headers = {"User-Agent": f"ImpactVizor (mailto:{self.email})"}
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the OpenAlex API with rate limiting"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Add polite pool parameter
            params['mailto'] = self.email
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            # Respect rate limits
            time.sleep(1)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to OpenAlex: {e}")
            # Additional error handling to help debug
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text[:200]}...")
            return None
    
    def search_works(self, query, filter_field=None, filter_value=None, page=1, per_page=25, additional_filters=None):
        """
        Search for works in OpenAlex
        
        Args:
            query (str): Search query
            filter_field (str, optional): Field to filter on
            filter_value (str, optional): Value for the filter
            page (int): Page number
            per_page (int): Results per page
            additional_filters (dict, optional): Additional filter parameters
            
        Returns:
            dict: Search results
        """
        params = {
            'search': query,
            'page': page,
            'per-page': per_page
        }
        
        # Initialize filter list
        filters = []
        
        # Handle date filter with the correct format for OpenAlex
        if filter_field == "publication_date" and filter_value:
            # Extract dates from format "start_date:end_date"
            dates = filter_value.split(":")
            if len(dates) == 2:
                start_date, end_date = dates
                
                # Ensure dates are formatted correctly (YYYY-MM-DD)
                try:
                    from datetime import datetime
                    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    # Use proper filter format for OpenAlex API with validated dates
                    formatted_start = start_date_obj.strftime("%Y-%m-%d")
                    formatted_end = end_date_obj.strftime("%Y-%m-%d")
                    # Split into from_date and to_date filters as separate params
                    filters.append(f"from_publication_date:{formatted_start}")
                    filters.append(f"to_publication_date:{formatted_end}")
                except ValueError as e:
                    print(f"Error formatting dates for OpenAlex: {e}")
                    # Use a default date range for fallback (last 10 years)
                    current_year = datetime.now().year
                    filters.append(f"from_publication_date:{current_year-10}-01-01")
                    filters.append(f"to_publication_date:{current_year}-12-31")
            else:
                print(f"Warning: Invalid date format for OpenAlex: {filter_value}")
                # Use a default date range for fallback (last 10 years)
                from datetime import datetime
                current_year = datetime.now().year
                filters.append(f"from_publication_date:{current_year-10}-01-01")
                filters.append(f"to_publication_date:{current_year}-12-31")
        elif filter_field and filter_value:
            # General filter format for other fields
            filters.append(f"{filter_field}:{filter_value}")
        
        # Add additional filters if provided
        if additional_filters and isinstance(additional_filters, dict):
            for field, value in additional_filters.items():
                if field == "publication_date" and ":" in value:
                    # Handle date range filters
                    dates = value.split(":")
                    if len(dates) == 2:
                        start_date, end_date = dates
                        try:
                            from datetime import datetime
                            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                            formatted_start = start_date_obj.strftime("%Y-%m-%d")
                            formatted_end = end_date_obj.strftime("%Y-%m-%d")
                            # Split into from_date and to_date filters as separate params
                            filters.append(f"from_publication_date:{formatted_start}")
                            filters.append(f"to_publication_date:{formatted_end}")
                        except ValueError as e:
                            print(f"Error formatting additional date filter for OpenAlex: {e}")
                else:
                    filters.append(f"{field}:{value}")
        
        # Combine all filters with AND operator
        if filters:
            params['filter'] = ','.join(filters)
        
        # Print API request parameters for debugging
        print(f"OpenAlex API request parameters: {params}")
        
        return self._make_request('works', params)
    
    def get_work_by_doi(self, doi):
        """
        Get work details by DOI
        
        Args:
            doi (str): DOI of the work
            
        Returns:
            dict: Work details
        """
        # Clean the DOI
        doi = doi.strip()
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')
        
        try:
            # Use urllib.parse.quote instead of requests.utils.quote
            from urllib.parse import quote
            encoded_doi = quote(doi)
            return self._make_request(f'works/https://doi.org/{encoded_doi}')
        except Exception as e:
            print(f"Error encoding DOI: {e}")
            # Try with unencoded DOI as fallback
            return self._make_request(f'works/https://doi.org/{doi}')
    
    def get_author(self, author_id):
        """
        Get author details
        
        Args:
            author_id (str): OpenAlex author ID
            
        Returns:
            dict: Author details
        """
        return self._make_request(f'authors/{author_id}')
    
    def get_institution(self, institution_id):
        """
        Get institution details
        
        Args:
            institution_id (str): OpenAlex institution ID
            
        Returns:
            dict: Institution details
        """
        return self._make_request(f'institutions/{institution_id}')
    
    def get_concept(self, concept_id):
        """
        Get concept details
        
        Args:
            concept_id (str): OpenAlex concept ID
            
        Returns:
            dict: Concept details
        """
        return self._make_request(f'concepts/{concept_id}')

class CrossrefClient:
    """Client for interacting with the Crossref API"""
    
    def __init__(self):
        self.base_url = "https://api.crossref.org"
        # Set user email for API etiquette
        self.email = os.getenv("USER_EMAIL", "user@example.com")
        self.headers = {"User-Agent": f"ImpactVizor (mailto:{self.email})"}
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the Crossref API with rate limiting"""
        if params is None:
            params = {}
        
        # Add email for polite pool
        params['mailto'] = self.email
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            # Respect rate limits
            time.sleep(1)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Crossref: {e}")
            # Additional error handling to help debug
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text[:200]}...")
            return None
    
    def search_works(self, query, from_date, until_date, rows=20, offset=0):
        """
        Search for works in Crossref
        
        Args:
            query (str): Search query
            from_date (str): Start date (YYYY-MM-DD)
            until_date (str): End date (YYYY-MM-DD)
            rows (int): Number of results to return
            offset (int): Result offset for pagination
            
        Returns:
            dict: Search results
        """
        # Use correct date format for Crossref API
        # Convert dates to the format Crossref accepts
        try:
            from_year = from_date.split('-')[0]
            until_year = until_date.split('-')[0]
            
            params = {
                'query': query,
                'filter': f'from-pub-date:{from_year},until-pub-date:{until_year}',
                'rows': rows,
                'offset': offset
            }
            
            return self._make_request('works', params)
        except Exception as e:
            print(f"Error formatting date parameters for Crossref: {e}")
            # Fallback to basic query without date filters
            return self._make_request('works', {'query': query, 'rows': rows, 'offset': offset})
    
    def search_by_author(self, author, from_date, until_date, rows=20, offset=0):
        """
        Search for works by a specific author
        
        Args:
            author (str): Author name
            from_date (str): Start date (YYYY-MM-DD)
            until_date (str): End date (YYYY-MM-DD)
            rows (int): Number of results to return
            offset (int): Result offset for pagination
            
        Returns:
            dict: Search results
        """
        try:
            from_year = from_date.split('-')[0]
            until_year = until_date.split('-')[0]
            
            params = {
                'query.author': author,
                'filter': f'from-pub-date:{from_year},until-pub-date:{until_year}',
                'rows': rows,
                'offset': offset
            }
            
            return self._make_request('works', params)
        except Exception as e:
            print(f"Error formatting date parameters for Crossref author search: {e}")
            # Fallback to basic query without date filters
            return self._make_request('works', {'query.author': author, 'rows': rows, 'offset': offset})
    
    def search_by_journal(self, journal, from_date, until_date, rows=20, offset=0):
        """
        Search for works in a specific journal
        
        Args:
            journal (str): Journal title or ISSN
            from_date (str): Start date (YYYY-MM-DD)
            until_date (str): End date (YYYY-MM-DD)
            rows (int): Number of results to return
            offset (int): Result offset for pagination
            
        Returns:
            dict: Search results
        """
        try:
            from_year = from_date.split('-')[0]
            until_year = until_date.split('-')[0]
            
            # Check if input is an ISSN
            if len(journal.replace('-', '')) == 8 and (journal.replace('-', '')).isdigit():
                params = {
                    'issn': journal,
                    'filter': f'from-pub-date:{from_year},until-pub-date:{until_year}',
                    'rows': rows,
                    'offset': offset
                }
            else:
                params = {
                    'query.container-title': journal,
                    'filter': f'from-pub-date:{from_year},until-pub-date:{until_year}',
                    'rows': rows,
                    'offset': offset
                }
            
            return self._make_request('works', params)
        except Exception as e:
            print(f"Error formatting date parameters for Crossref journal search: {e}")
            # Fallback to basic query without date filters
            if len(journal.replace('-', '')) == 8 and (journal.replace('-', '')).isdigit():
                return self._make_request('works', {'issn': journal, 'rows': rows, 'offset': offset})
            else:
                return self._make_request('works', {'query.container-title': journal, 'rows': rows, 'offset': offset})
    
    def get_work_by_doi(self, doi):
        """
        Get work details by DOI
        
        Args:
            doi (str): DOI of the work
            
        Returns:
            dict: Work details
        """
        # Clean the DOI
        doi = doi.strip()
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')
        
        try:
            # Use urllib.parse.quote instead of requests.utils.quote
            from urllib.parse import quote
            encoded_doi = quote(doi)
            return self._make_request(f'works/{encoded_doi}')
        except Exception as e:
            print(f"Error encoding DOI for Crossref: {e}")
            # Try with unencoded DOI as fallback
            return self._make_request(f'works/{doi}')
