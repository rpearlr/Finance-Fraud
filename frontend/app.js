let API_BASE = "";
if (API_BASE.endsWith('/')) {
  API_BASE = API_BASE.slice(0, -1);
}
// Global helper to send a query to the AI Agent endpoint
async function sendPrompt(query, overrideAgent = null) {
  try {
    const payload = { question: query };
    if (overrideAgent) {
      payload.agent_override = overrideAgent;
    }
    const response = await fetch(`${API_BASE}/agent/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error communicating with AI agent:", error);
    return {
      answer: "Sorry, I couldn't connect to the backend. Please ensure the FastAPI server is running on http://localhost:8000.",
      agent_used: "error",
      latency_ms: 0
    };
  }
}

// Ingest endpoint caller for new data
async function ingestTransaction(transaction) {
  try {
    const payload = {
      data_type: "transaction",
      transaction: transaction
    };
    const response = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error ingesting transaction:", error);
    return null;
  }
}

// Predict endpoint caller for the fraud detection page
async function predictFraud(features) {
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error scoring transaction:", error);
    return null;
  }
}

// Search endpoint caller for the regulatory search page
async function searchRegulatory(query, top_k = 5) {
  try {
    const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&top_k=${top_k}`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Error searching regulations:", error);
    return null;
  }
}

// Upload custom portfolio data for Holt-Winters forecast
async function uploadPortfolioData(file, days = 30) {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/forecast/custom?days=${days}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Error generating custom forecast:", error);
    return null;
  }
}

// Wait for DOM
document.addEventListener("DOMContentLoaded", () => {
  console.log("FinRisk Frontend initialized");

  // ── Auth Guard ──────────────────────────────────────────
  const token = localStorage.getItem('finrisk_token');
  const path = window.location.pathname;
  
  // If no token and not on login page, redirect to root (login)
  if (!token && path !== '/' && !path.includes('login')) {
    window.location.href = '/';
    return;
  }

  // Update username in UI if exists
  const userNameElem = document.getElementById('user-profile-name');
  if (userNameElem && localStorage.getItem('finrisk_user')) {
    userNameElem.textContent = localStorage.getItem('finrisk_user');
  }

  // ── Theme management ──────────────────────────────────────
  const savedTheme = localStorage.getItem('finrisk-theme');
  if (savedTheme === 'dark') {
    document.documentElement.classList.add('dark');
  }
  updateThemeIcon();

  // ── Sidebar collapse ─────────────────────────────────────
  const savedSidebar = localStorage.getItem('finrisk-sidebar');
  if (savedSidebar === 'collapsed') {
    const sidebar = document.querySelector('.sidebar');
    const toggle = document.querySelector('.sidebar-toggle');
    if (sidebar) sidebar.classList.add('collapsed');
    if (toggle) toggle.style.width = 'var(--sidebar-collapsed-w)';
  }
});

function handleLogout() {
  localStorage.removeItem('finrisk_token');
  localStorage.removeItem('finrisk_user');
  // Optional: call backend logout
  fetch(`${API_BASE}/auth/logout`, { method: 'POST' }).finally(() => {
    window.location.href = '/';
  });
}
window.handleLogout = handleLogout;

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('finrisk-theme', isDark ? 'dark' : 'light');
  updateThemeIcon();
}

function updateThemeIcon() {
  const btn = document.getElementById('theme-btn');
  if (!btn) return;
  const isDark = document.documentElement.classList.contains('dark');
  btn.innerHTML = isDark
    ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`
    : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
}

function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const toggle = document.querySelector('.sidebar-toggle');
  if (!sidebar) return;
  sidebar.classList.toggle('collapsed');
  const isCollapsed = sidebar.classList.contains('collapsed');
  if (toggle) toggle.style.width = isCollapsed ? 'var(--sidebar-collapsed-w)' : 'var(--sidebar-w)';
  localStorage.setItem('finrisk-sidebar', isCollapsed ? 'collapsed' : 'expanded');
}

function suggest(text) {
  const input = document.getElementById('chatInput');
  if (input) { input.value = text; input.focus(); }
}

async function fetchDashboardKPIs() {
  try {
    const response = await fetch(`${API_BASE}/dashboard/kpis`);
    if (!response.ok) throw new Error("Failed to fetch KPIs");

    const data = await response.json();

    // Top KPIs
    if (document.getElementById('kpi-tx-today')) {
      document.getElementById('kpi-tx-today').innerText = data.transactions_today.toLocaleString();
    }
    if (document.getElementById('kpi-fraud-flagged')) {
      document.getElementById('kpi-fraud-flagged').innerText = data.fraud_flagged.toLocaleString();
    }
    if (document.getElementById('kpi-avg-score')) {
      document.getElementById('kpi-avg-score').innerText = data.avg_fraud_score.toFixed(3);
    }
    if (document.getElementById('kpi-portfolio')) {
      // e.g. 4820000000 -> 482
      const cr = (data.portfolio_value / 10000000).toFixed(0);
      document.getElementById('kpi-portfolio').innerText = `₹${cr} Cr`;
    }

    // Active agents
    if (document.getElementById('kpi-active-agents')) {
      document.getElementById('kpi-active-agents').innerText = data.active_agents;
    }

    // Hourly chart
    const chartContainer = document.getElementById('hourly-chart-container');
    if (chartContainer && data.hourly_fraud && data.hourly_fraud.length > 0) {
      chartContainer.innerHTML = '';
      data.hourly_fraud.forEach(item => {
        // Convert '14' to '02 PM'
        let hrNum = parseInt(item.hour);
        let ampm = hrNum >= 12 ? 'PM' : 'AM';
        let hr12 = hrNum % 12;
        if (hr12 === 0) hr12 = 12;
        let label = `${hr12.toString().padStart(2, '0')} ${ampm}`;

        let score = parseFloat(item.avg_score);
        let pct = Math.min(100, Math.max(5, score * 100)); // cap width for display

        let colorClass = 'var(--brand-green)';
        if (score >= 0.5) colorClass = 'var(--amber)';
        if (score >= 0.8) colorClass = 'var(--danger-red)';

        chartContainer.innerHTML += `
          <div class="bar-row" style="display: flex; align-items: center; gap: 12px; width: 100%;">
            <div class="bar-label" style="width: 50px; font-size: 11px; font-family: var(--fm); color: var(--tx2);">${label}</div>
            <div class="bar-track" style="flex: 1; height: 8px; background: var(--sur2); border-radius: 4px; overflow: hidden;">
              <div class="bar-fill" style="height: 100%; width: ${pct}%; background-color: ${colorClass}; border-radius: 4px; transition: width 1s var(--ease);"></div>
            </div>
            <div class="bar-val" style="width: 30px; font-size: 12px; font-weight: 500; text-align: right; color: var(--tx);">${score.toFixed(2)}</div>
          </div>
        `;
      });
    }

    // Recent alerts
    const alertsContainer = document.getElementById('recent-alerts-container');
    if (alertsContainer && data.recent_alerts && data.recent_alerts.length > 0) {
      alertsContainer.innerHTML = '';
      data.recent_alerts.forEach(alert => {
        const amtStr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(alert.amount);
        let badgeClass = alert.score >= 0.8 ? 'badge-high' : (alert.score >= 0.5 ? 'badge-med' : 'badge-low');
        let badgeText = alert.score >= 0.8 ? 'HIGH' : (alert.score >= 0.5 ? 'MED' : 'LOW');

        alertsContainer.innerHTML += `
        <div class="card flex-row" style="justify-content: space-between;">
          <div class="flex-row">
            <span class="badge badge-pill ${badgeClass}">${badgeText}</span>
            <div>
              <div class="weight-500">${alert.tx_id}</div>
              <div class="muted" style="font-size: 11px;">${amtStr} &middot; ${alert.time} &middot; Score: ${alert.score}</div>
            </div>
          </div>
        </div>
        `;
      });
    }
  } catch (err) {
    console.error("Dashboard fetch error:", err);
  }
}

// ── Animation Helpers ─────────────────────────────────────

function initSparklines() {
  const dataMap = {
    'spark-tx': [10, 15, 12, 18, 24, 20, 25, 22, 30, 28, 35],
    'spark-fraud': [2, 1, 3, 2, 5, 3, 2, 4, 2, 1, 3],
    'spark-score': [0.1, 0.12, 0.08, 0.15, 0.14, 0.11, 0.09, 0.12, 0.1, 0.08, 0.09],
    'spark-port': [450, 452, 448, 455, 460, 458, 465, 470, 468, 475, 482],
    'spark-forecast': [482, 485, 486, 488, 490, 491, 493, 495, 496, 498, 501],
    'spark-sharpe': [1.35, 1.36, 1.37, 1.38, 1.36, 1.39, 1.40, 1.41, 1.42, 1.41, 1.42],
    'spark-drawdown': [-4.1, -4.2, -4.5, -4.4, -4.6, -4.5, -4.4, -4.3, -4.2, -4.3, -4.3]
  };

  const colors = {
    'spark-tx': 'var(--brand)',
    'spark-fraud': 'var(--danger-red)',
    'spark-score': 'var(--brand)',
    'spark-port': 'var(--brand)',
    'spark-forecast': 'var(--violet)',
    'spark-sharpe': 'var(--brand-green)',
    'spark-drawdown': 'var(--danger-red)'
  };

  Object.keys(dataMap).forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    
    const data = dataMap[id];
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    
    let pathD = "";
    data.forEach((val, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((val - min) / range) * 80 - 10; // 10% padding
      pathD += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });

    el.innerHTML = `
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <path class="spark-path path-draw" d="${pathD}" stroke="${colors[id]}" />
      </svg>
    `;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    initSparklines();
  }, 300);
});

// Generate PDF Report caller
async function generatePDFReport(agent, query, txId = null) {
  try {
    let report_type = agent === 'fraud_expert_agent' ? 'fraud' : 'investment';
    let payload = { report_type: report_type, query: query };
    if (report_type === 'fraud') {
       payload.tx_id = txId || 'TX-999999';
    }

    const response = await fetch(`${API_BASE}/report/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    return { pdf_url: url };
  } catch (error) {
    console.error("Error generating PDF:", error);
    return null;
  }
}

