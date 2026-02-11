// Extension loaded - immediate log
console.log("🤖 Delta Bot Extension: Script loaded on", window.location.href);

// Wait for page to be ready
function initExtension() {
  console.log("🤖 Delta Bot Extension: Initializing...");

  // Create the overlay panel
  const panel = document.createElement('div');
  panel.id = 'delta-bot-overlay';
  panel.innerHTML = `
    <div class="header">
      <span>🤖 Delta Bot</span>
      <div class="header-actions">
        <button id="refresh-bot" title="Refresh Data">↻</button>
        <button id="minimize-bot" title="Minimize">—</button>
      </div>
    </div>
    <div class="content">
      <div class="stat-row">
        <span class="label">Bot Signal:</span>
        <span id="bot-signal" class="value neutral">None</span>
      </div>
      <div class="stat-row">
        <span class="label">Trend (15m):</span>
        <span id="bot-trend" class="value">Loading...</span>
      </div>
      <div class="stat-row">
        <span class="label">RSI (5m):</span>
        <span id="bot-rsi" class="value">--</span>
      </div>
      <div class="stat-row">
        <span class="label">Market Bias:</span>
        <span id="bot-bias" class="value">Loading...</span>
      </div>
      <div class="stat-row">
        <span class="label">Last Event:</span>
        <span id="bot-event" class="value">--</span>
      </div>
      <div class="stat-row">
        <span class="label">Liquidity:</span>
        <span id="bot-liquidity" class="value">None</span>
      </div>
      <div class="section-title">Patterns Detected</div>
      <div id="bot-patterns" class="patterns-list">
        None
      </div>
      <div class="section-title">Key Levels</div>
      <div class="stat-row">
        <span class="label">Support:</span>
        <span id="bot-support" class="value">--</span>
      </div>
      <div class="stat-row">
        <span class="label">Resistance:</span>
        <span id="bot-resistance" class="value">--</span>
      </div>
      <div class="stat-row">
        <span class="label">Last Price:</span>
        <span id="bot-price" class="value">--</span>
      </div>
      <div id="stale-warning" class="warning-row" style="display: none;">
        ⚠️ Data may be stale
      </div>
      <div class="footer">
        <span class="label">Bot Status:</span>
        <span id="bot-status" class="value" style="font-size: 10px; color: #00e676;">Initializing...</span>
      </div>
      <div class="footer" style="border-top: none; margin-top: 0;">
        <span class="label">Last Bot Update:</span>
        <span id="bot-heartbeat" class="value">--</span>
      </div>
    </div>
  `;

  document.body.appendChild(panel);
  console.log("🤖 Delta Bot Extension: ✅ Panel injected successfully!");

  // --- DRAG FUNCTIONALITY ---
  const header = panel.querySelector('.header');
  let isDragging = false;
  let currentX;
  let currentY;
  let initialX;
  let initialY;
  let xOffset = 0;
  let yOffset = 0;

  header.addEventListener("mousedown", dragStart);
  document.addEventListener("mouseup", dragEnd);
  document.addEventListener("mousemove", drag);

  function dragStart(e) {
    initialX = e.clientX - xOffset;
    initialY = e.clientY - yOffset;

    // Only drag if clicking the header (but not the minimize button)
    if (e.target.closest('.header') && !e.target.closest('button')) {
      isDragging = true;
    }
  }

  function dragEnd(e) {
    initialX = currentX;
    initialY = currentY;
    isDragging = false;
  }

  function drag(e) {
    if (isDragging) {
      e.preventDefault();
      currentX = e.clientX - initialX;
      currentY = e.clientY - initialY;

      xOffset = currentX;
      yOffset = currentY;

      setTranslate(currentX, currentY, panel);
    }
  }

  function setTranslate(xPos, yPos, el) {
    el.style.transform = `translate3d(${xPos}px, ${yPos}px, 0)`;
  }
  // --------------------------

  // Refresh button
  const refreshBtn = panel.querySelector('#refresh-bot');
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.classList.add('rotating');
    await fetchAnalysis();
    setTimeout(() => refreshBtn.classList.remove('rotating'), 500);
  });

  // Minimize functionality
  const contentDiv = panel.querySelector('.content');
  const minimizeBtn = panel.querySelector('#minimize-bot');
  minimizeBtn.addEventListener('click', () => {
    if (contentDiv.style.display === 'none') {
      contentDiv.style.display = 'block';
    } else {
      contentDiv.style.display = 'none';
    }
  });

  // Start polling
  setInterval(fetchAnalysis, 1000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initExtension);
} else {
  initExtension();
}

let lastSignaledSymbol = null;

async function sendPrioritySignal(symbol) {
  if (symbol === lastSignaledSymbol) return;
  chrome.runtime.sendMessage({ type: 'SET_PRIORITY', symbol: symbol }, (response) => {
    if (response && response.success) {
      lastSignaledSymbol = symbol;
      console.log("⚡ Signal sent for priority:", symbol);
    } else {
      console.error("❌ Failed to send priority signal:", response ? response.error : 'No response');
    }
  });
}

// Polling function
async function fetchAnalysis() {
  // Guard against invalidated context (happens after extension reload)
  if (!chrome.runtime?.id) {
    console.warn("🤖 Delta Bot: Extension context invalidated. Refresh the page.");
    return;
  }

  chrome.runtime.sendMessage({ type: 'FETCH_BOT_DATA' }, async (response) => {
    if (!response || !response.success) {
      console.warn("🤖 Delta Bot: Connection failed - Bot might be offline or starting.");
      document.getElementById('bot-trend').innerText = "Bot Offline";
      document.getElementById('bot-trend').className = 'value down';
      return;
    }
    const data = response.data;

    if (data.error) {
      document.getElementById('bot-trend').innerText = "Waiting for bot...";
      return;
    }

    // Determine active symbol on page (simplistic check for now, or just show whatever the bot scans)
    // Ideally, current page URL/DOM validation would happen here.
    // The bot currently returns a dictionary of ALL scanned symbols if we modified it? 
    // Wait, bot_pro.py returns `bot_instance.latest_analysis`.
    // The bot scans sequentially. `latest_analysis` is a dict {symbol: data}.
    // We need to find the matching symbol for the page.

    // Extract symbol from URL (e.g., https://india.delta.exchange/app/trade/BTCUSD)
    // URL format can be: .../trade/BTC/BTCUSD or .../trade/BTCUSD
    const url = window.location.href;
    const urlParts = url.split('/');
    let pageSymbol = urlParts[urlParts.length - 1]; // Try last part first

    // Clean up symbol (remove query params if any)
    if (pageSymbol.includes('?')) pageSymbol = pageSymbol.split('?')[0];

    // Fallback: Check common pairs if URL structure is weird
    if (!pageSymbol || pageSymbol.length < 3) {
      // Try finding a known symbol in the URL
      const knownSymbols = Object.keys(data);
      pageSymbol = knownSymbols.find(s => url.includes(s)) || 'BTCUSD';
    }

    console.log("🤖 Delta Bot: Detected page symbol:", pageSymbol);

    // Signal priority to bot
    await sendPrioritySignal(pageSymbol);

    let activeData = data[pageSymbol];

    // If exact match fails, try fuzzy match (e.g. data has BTCUSD but url has BTC)
    if (!activeData) {
      const key = Object.keys(data).find(k => k.includes(pageSymbol) || pageSymbol.includes(k));
      if (key) activeData = data[key];
    }

    if (!activeData) {
      document.getElementById('bot-trend').innerText = `No Data for ${pageSymbol}`;
      document.getElementById('bot-trend').className = 'value neutral';
      // Clear other fields
      document.getElementById('bot-rsi').innerText = "--";
      document.getElementById('bot-patterns').innerText = "Scanning...";
      document.getElementById('bot-support').innerText = "--";
      document.getElementById('bot-resistance').innerText = "--";
      document.getElementById('bot-price').innerText = "--";
      return;
    }

    // Update UI headers to show which coin we are displaying
    const panel = document.getElementById('delta-bot-overlay');
    if (panel) {
      const headerTitle = panel.querySelector('.header span');
      if (headerTitle) headerTitle.innerText = `🤖 ${pageSymbol}`;
    }

    // Update Signal
    const signalEl = document.getElementById('bot-signal');
    const signalVal = activeData.ut_signal || "None";
    signalEl.innerText = signalVal;
    signalEl.className = 'value ' + (signalVal.includes('BUY') ? 'up' : signalVal.includes('SELL') ? 'down' : 'neutral');

    // Update UI
    const trendEl = document.getElementById('bot-trend');
    const trendVal = activeData.trend_15m || "neutral";
    trendEl.innerText = trendVal.toUpperCase();
    trendEl.className = 'value ' + (trendVal === 'bullish' || trendVal === 'up' ? 'up' : trendVal === 'bearish' || trendVal === 'down' ? 'down' : 'neutral');

    const biasEl = document.getElementById('bot-bias');
    const biasVal = activeData.market_bias || "neutral";
    biasEl.innerText = biasVal.toUpperCase();
    biasEl.className = 'value ' + (biasVal === 'bullish' ? 'up' : biasVal === 'bearish' ? 'down' : 'neutral');

    const rsiEl = document.getElementById('bot-rsi');
    rsiEl.innerText = activeData.rsi ? parseFloat(activeData.rsi).toFixed(1) : "--";

    const eventEl = document.getElementById('bot-event');
    eventEl.innerText = activeData.last_event || "None";

    const sweepEl = document.getElementById('bot-liquidity'); // Renamed from liqEl to sweepEl
    if (activeData.sweeps && activeData.sweeps.length > 0) {
      sweepEl.innerText = activeData.sweeps.join(', ');
      sweepEl.className = 'value up';
    } else {
      sweepEl.innerText = "None";
      sweepEl.className = 'value';
    }

    const patternsEl = document.getElementById('bot-patterns');
    if (activeData.patterns && activeData.patterns.length > 0) {
      // Check if it's the new object format or old string format (backward compatibility)
      patternsEl.innerHTML = activeData.patterns.map(p => {
        if (typeof p === 'object') {
          // Parse time (assuming ISO or similar string)
          const date = new Date(p.time);
          const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          return `<div class="badge-row"><span class="badge">${p.name}</span> <span class="badge-detail">${p.price.toFixed(2)} | ${timeStr}</span></div>`;
        } else {
          return `<span class="badge">${p}</span>`;
        }
      }).join('');
    } else {
      patternsEl.innerText = "No matches";
    }

    document.getElementById('bot-support').innerText = parseFloat(activeData.support).toFixed(2);
    document.getElementById('bot-resistance').innerText = parseFloat(activeData.resistance).toFixed(2);
    document.getElementById('bot-price').innerText = parseFloat(activeData.price).toFixed(2);

    // Update Status
    const statusEl = document.getElementById('bot-status');
    statusEl.innerText = data["status"] || "Active";

    // Update Heartbeat
    const heartbeatEl = document.getElementById('bot-heartbeat');
    const heartbeat = data["_bot_heartbeat"] || "--";
    heartbeatEl.innerText = heartbeat.split(' ')[1] || heartbeat; // Show only time if possible

    // Stale detection
    if (data["_bot_heartbeat"]) {
      const lastUpdate = new Date(data["_bot_heartbeat"]);
      const now = new Date();
      const diffSec = (now - lastUpdate) / 1000;
      const warningEl = document.getElementById('stale-warning');
      if (diffSec > 60) {
        warningEl.style.display = 'block';
        heartbeatEl.style.color = '#ff3d00';
      } else {
        warningEl.style.display = 'none';
        heartbeatEl.style.color = '#a0a0b0';
      }
    }

  });
}
