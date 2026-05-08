const API_BASE = "";

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
          <div class="bar-row">
            <div class="bar-label">${label}</div>
            <div class="bar-track"><div class="bar-fill" style="width: ${pct}%; background-color: ${colorClass};"></div></div>
            <div class="bar-val">${score.toFixed(2)}</div>
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

