import requests
import time

def trigger_scraping_channels(api_key, channels, count, start_date, end_date, sort_by, search_term):
    """Triggers the Bright Data collector for YouTube channels."""
    url = "https://api.brightdata.com/dca/trigger" # Standard Bright Data API endpoint
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # This payload structure depends on your specific Bright Data Collector setup
    payload = {
        "channels": channels,
        "count": count,
        "start_date": start_date,
        "end_date": end_date,
        "sort_by": sort_by
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json() # Returns {'snapshot_id': 's_xxxxxx'}

def get_progress(api_key, snapshot_id):
    """Checks if the scraping job is finished."""
    url = f"https://api.brightdata.com/dca/snapshot/{snapshot_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    return response.json() # Returns {'status': 'ready' or 'running'}

def get_output(api_key, snapshot_id, format="json"):
    """Downloads the final scraped data."""
    url = f"https://api.brightdata.com/dca/snapshot/{snapshot_id}/download?format={format}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    return response.json()