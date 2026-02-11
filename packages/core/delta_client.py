import time
import hmac
import hashlib
import requests
import json
import urllib.parse

class DeltaClient:
    def __init__(self, api_key, api_secret, base_url="https://api.india.delta.exchange"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "delta-bot-v1",
            "Content-Type": "application/json"
        })

    def _generate_signature(self, method, path, query_string, payload, timestamp):
        # Signature = HMAC-SHA256(SECRET, METHOD + TIMESTAMP + PATH + QUERY_STRING + BODY)
        body = json.dumps(payload) if payload else ""
        message = f"{method}{timestamp}{path}{query_string}{body}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method, endpoint, params=None, payload=None, auth=True, retries=3):
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(retries):
            try:
                timestamp = str(int(time.time()))
                
                # Prepare headers
                headers = {}
                if auth:
                    # Query string handling for signature
                    query_string = ""
                    if params:
                        query_string = urllib.parse.urlencode(params)
                    
                    signature = self._generate_signature(
                        method.upper(), endpoint, query_string, payload, timestamp
                    )
                    headers.update({
                        "api-key": self.api_key,
                        "signature": signature,
                        "timestamp": timestamp
                    })

                response = self.session.request(
                    method, url, params=params, json=payload, headers=headers, timeout=15
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    print(f"⚠️ Timeout on attempt {attempt + 1}/{retries}, retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"❌ Request timed out after {retries} attempts")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                # Don't retry on client errors (4xx), only server errors (5xx)
                if e.response is not None and 500 <= e.response.status_code < 600:
                    if attempt < retries - 1:
                        print(f"⚠️ Server error {e.response.status_code}, retrying...")
                        time.sleep(2)
                        continue
                print(f"❌ API Error: {e}")
                if e.response is not None:
                    print(f"Response: {e.response.text}")
                return None
                
            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    print(f"⚠️ Connection error, retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"❌ Connection failed after {retries} attempts: {e}")
                    return None
                    
            except Exception as e:
                print(f"❌ Request Failed: {e}")
                return None
        
        return None

    def get_products(self):
        """Fetch all available products to map symbols to IDs."""
        resp = self._request("GET", "/v2/products", auth=False)
        return resp.get('result', []) if resp else []

    def get_candles(self, symbol, resolution="1m", start=None, end=None):
        """
        Fetch historical candles.
        resolution: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
        """
        params = {
            "symbol": symbol,
            "resolution": resolution
        }
        if start: params["start"] = start
        if end: params["end"] = end
        
        resp = self._request("GET", "/v2/history/candles", params=params, auth=False)
        return resp.get('result', []) if resp else []

    def place_order(self, product_id, size, side, price=None, order_type="limit_order"):
        """
        Place a new order.
        side: "buy" or "sell"
        order_type: "limit_order", "market_order", "stop_order"
        """
        payload = {
            "product_id": int(product_id),
            "size": int(size),
            "side": side.lower(),
            "order_type": order_type
        }
        if price:
            payload["limit_price"] = str(price)
            
        resp = self._request("POST", "/v2/orders", payload=payload, auth=True)
        return resp.get('result', {}) if resp else None

    def set_leverage(self, product_id, leverage):
        """Set leverage for a specific product."""
        payload = {
            "leverage": str(leverage)
        }
        return self._request("POST", f"/v2/products/{product_id}/orders/leverage", payload=payload, auth=True)

    def get_positions(self):
        """Get current open positions."""
        resp = self._request("GET", "/v2/positions", auth=True)
        return resp.get('result', []) if resp else []

    def get_ticker(self, symbol):
        """Get 24hr ticker stats for a symbol."""
        resp = self._request("GET", f"/v2/tickers/{symbol}", auth=False)
        return resp.get('result', {}) if resp else None
