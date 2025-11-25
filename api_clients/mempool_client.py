from dotenv import load_dotenv
import os
import httpx
from typing import Dict, Any, List, Optional
from ratelimit import limits, sleep_and_retry
import time

load_dotenv()
MEMPOOL_BASE_URL = os.getenv("MEMPOOL_BASE_URL")
RATE_LIMIT = os.getenv("MEMPOOL_RATE_LIMIT")
RATE_LIMIT_INTERVAL = os.getenv("MEMPOOL_RATE_LIMIT_INTERVAL")
MAX_RETRIES = os.getenv("MAX_RETRIES")

class MempoolClient:
    """Client for Mempool.space API.
    
    Docs: https://mempool.space/docs/api/rest
    """
    
    def __init__(self, base_url: str = MEMPOOL_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)


    @sleep_and_retry
    @limits(calls=RATE_LIMIT, period=RATE_LIMIT_INTERVAL)
    def _rate_limited_get(self, url: str, max_retries: int = MAX_RETRIES):
        """Rate-limited HTTP GET with 429 handling"""
        for attempt in range(max_retries):
            try:
                response = self.client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                
                print(f"Rate limited (429). Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise  # Re-raise non-429 errors
    
        raise Exception(f"Failed after {max_retries} retries due to rate limiting")
    

    def get_transaction(self, txid: str) -> Dict[str, Any]:
        """Get transaction by txid.
        
        GET /tx/:txid
        """
        url = f"{self.base_url}/tx/{txid}"
        
        return self._rate_limited_get(url)
    
    def get_address_txs(self, address: str) -> List[Dict[str, Any]]:
        """Get transaction history for address.
        
        GET /address/:address/txs
        """
        # TODO: Implement
        pass
    
    def close(self):
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

