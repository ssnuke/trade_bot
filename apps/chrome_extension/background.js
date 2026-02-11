// Background Service Worker for Delta Bot
// Handles API requests to avoid Mixed Content (HTTPS -> HTTP) blocks

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'FETCH_BOT_DATA') {
        fetch('http://127.0.0.1:5005/analysis')
            .then(async response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    const text = await response.text();
                    throw new Error("Bot returned non-JSON response (Maybe wrong port?)");
                }
                return response.json();
            })
            .then(data => sendResponse({ success: true, data: data }))
            .catch(error => {
                console.error('Fetch error:', error);
                sendResponse({ success: false, error: error.message });
            });
        return true; // Keep channel open for async response
    }

    if (request.type === 'SET_PRIORITY') {
        fetch('http://127.0.0.1:5005/set_priority', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: request.symbol })
        })
            .then(response => response.json())
            .then(data => sendResponse({ success: true, data: data }))
            .catch(error => sendResponse({ success: false, error: error.message }));
        return true;
    }
});
