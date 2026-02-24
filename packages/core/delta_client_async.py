import asyncio
import hmac
import hashlib
import json
import urllib.parse
from typing import Optional, Any
import aiohttp


class AsyncDeltaClient:
    """Async version of Delta Exchange API client."""
    
    def __init__(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        base_url: str = "https://api.india.delta.exchange",
        timeout: int = 15,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

    def _generate_signature(
        self,
        method: str,
        path: str,
        query_string: str,
        payload: Optional[dict],
        timestamp: str
    ) -> str:
        body = json.dumps(payload) if payload else ""
        message = f"{method}{timestamp}{path}{query_string}{body}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        payload: Optional[dict] = None,
        auth: bool = True
    ) -> Optional[dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                timestamp = str(int(asyncio.get_event_loop().time()))
                
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if auth:
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

                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.request(
                        method,
                        url,
                        params=params,
                        json=payload,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
                        
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Timeout on attempt {attempt + 1}/{self.max_retries}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"❌ Request timed out after {self.max_retries} attempts")
                    return None
                    
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Client error, retrying: {e}")
                    await asyncio.sleep(2)
                    continue
                print(f"❌ API Error: {e}")
                return None
                    
            except Exception as e:
                print(f"❌ Request Failed: {e}")
                return None
        
        return None

    async def get_products(self) -> list[dict[str, Any]]:
        resp = await self._request("GET", "/v2/products", auth=False)
        return resp.get('result', []) if resp else []

    async def get_candles(
        self,
        symbol: str,
        resolution: str = "1m",
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "resolution": resolution
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        resp = await self._request("GET", "/v2/history/candles", params=params, auth=False)
        return resp.get('result', []) if resp else []

    async def place_order(
        self,
        product_id: int,
        size: float,
        side: str,
        price: Optional[float] = None,
        order_type: str = "limit_order"
    ) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {
            "product_id": int(product_id),
            "size": int(size),
            "side": side.lower(),
            "order_type": order_type
        }
        if price:
            payload["limit_price"] = str(price)
            
        resp = await self._request("POST", "/v2/orders", payload=payload, auth=True)
        return resp.get('result', {}) if resp else None

    async def set_leverage(self, product_id: int, leverage: int) -> Optional[dict[str, Any]]:
        payload = {"leverage": str(leverage)}
        return await self._request(
            "POST",
            f"/v2/products/{product_id}/orders/leverage",
            payload=payload,
            auth=True
        )

    async def get_positions(self) -> list[dict[str, Any]]:
        resp = await self._request("GET", "/v2/positions", auth=True)
        return resp.get('result', []) if resp else []

    async def get_ticker(self, symbol: str) -> Optional[dict[str, Any]]:
        resp = await self._request("GET", f"/v2/tickers/{symbol}", auth=False)
        return resp.get('result', {}) if resp else None
