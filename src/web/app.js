/**
 * QuakeAI Frontend — Application Logic
 * =====================================
 * Handles tab switching, API calls, Chart.js rendering,
 * Leaflet map, prediction form, and analytics.
 */

// ─── API Base URL (environment-aware) ────────────────────────
// Mode A — Production / Docker: frontend served by Flask on same origin
//          → use relative paths (no CORS needed)
// Mode B — Development: Live Server (port 5500), file:// protocol, etc.
//          → point explicitly at Flask backend on port 5000
const API = (() => {
    const { protocol, hostname, port } = window.location;
    // file:// → opened HTML directly
    if (protocol === "file:") return "http://127.0.0.1:5000";
    // Same port as backend (Flask serves frontend) → relative
    if (port === "5000" || port === "") return "";
    // Any other port (e.g. Live Server 5500) → cross-origin to backend
    return `http://${hostname}:5000`;
})();

// ─── Chart.js Global Defaults ────────────────────────────────
Chart.defaults.color = "#737373";
Chart.defaults.borderColor = "#2d2d2d";
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 10;

// Region color map for consistent chart colors
const REGION_COLORS = {
    "Andaman & Nicobar": "#06b6d4",
    "Central": "#8b5cf6",
    "Central Asia": "#a855f7",
    "East": "#3b82f6",
    "Himalayan Belt": "#f43f5e",
    "Myanmar Region": "#ec4899",
    "North": "#22c55e",
    "Northeast": "#14b8a6",
    "Northwest": "#f59e0b",
    "Other": "#6b7280",
    "South": "#ef4444",
    "Southeast": "#e879f9",
    "Southwest": "#84cc16",
    "West": "#fb923c",
};

// Magnitude distribution bar colors
const MAG_DIST_COLORS = {
    "<2": "#22c55e",
    "2-3": "#86efac",
    "3-4": "#3b82f6",
    "4-5": "#f59e0b",
    "5-6": "#f97316",
    "6-7": "#ef4444",
    "7+": "#dc2626",
};

// Chart instance cache (for destroy/recreate)
const charts = {};

// ─── Tab Switching ───────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;

        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        document.getElementById(`panel-${tab}`).classList.add("active");

        // Lazy init
        if (tab === "map" && !mapState.initialized) initMap();
        if (tab === "analytics" && !analyticsLoaded) loadAnalytics();
    });
});

// ─── API Helpers ─────────────────────────────────────────────
async function apiFetch(path) {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) throw new Error(`API error ${res.status}`);
    return res.json();
}

// ─── Dashboard ───────────────────────────────────────────────
async function loadDashboard() {
    try {
        const [regionData, metrics, magDist] = await Promise.all([
            apiFetch("/analytics/region"),
            apiFetch("/model/metrics"),
            apiFetch("/analytics/magnitude_dist").catch(() => []),
        ]);

        // Metric cards
        const totalQuakes = regionData.reduce((s, r) => s + r.count, 0);
        const avgMag = totalQuakes > 0
            ? (regionData.reduce((s,r) => s + r.avg_magnitude * r.count, 0) / totalQuakes).toFixed(2)
            : "—";
        const maxMag = regionData.length > 0
            ? Math.max(...regionData.map(r => r.max_magnitude)).toFixed(1)
            : "—";

        document.getElementById("m-total").textContent = totalQuakes.toLocaleString();
        document.getElementById("m-avg-mag").textContent = avgMag;
        document.getElementById("m-max-mag").textContent = maxMag;
        document.getElementById("m-r2").textContent = metrics.r2;
        document.getElementById("m-rmse").textContent = metrics.rmse;

        // Region bar chart — earthquake count
        renderChart("regionBarChart", "bar", {
            labels: regionData.map(r => r.region),
            datasets: [{
                label: "Earthquake Count",
                data: regionData.map(r => r.count),
                backgroundColor: regionData.map(r => REGION_COLORS[r.region] || "#6b7280"),
                borderRadius: 4,
                borderSkipped: false,
            }]
        }, { indexAxis: "y" });

        // Region avg magnitude chart
        renderChart("regionMagChart", "bar", {
            labels: regionData.map(r => r.region),
            datasets: [{
                label: "Avg Magnitude",
                data: regionData.map(r => r.avg_magnitude),
                backgroundColor: regionData.map(r => {
                    const c = REGION_COLORS[r.region] || "#6b7280";
                    return c + "99"; // semi-transparent
                }),
                borderColor: regionData.map(r => REGION_COLORS[r.region] || "#6b7280"),
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
            }]
        });

        // Magnitude distribution chart (if data available)
        if (magDist && magDist.length > 0) {
            renderChart("magDistChart", "bar", {
                labels: magDist.map(d => d.range),
                datasets: [{
                    label: "Earthquake Count",
                    data: magDist.map(d => d.count),
                    backgroundColor: magDist.map(d => MAG_DIST_COLORS[d.range] || "#6b7280"),
                    borderRadius: 4,
                    borderSkipped: false,
                }]
            });
        }

        // Region table
        const tbody = document.getElementById("region-table-body");
        tbody.innerHTML = regionData.map(r => `
            <tr>
                <td><span style="color:${REGION_COLORS[r.region] || '#6b7280'}">●</span> ${r.region}</td>
                <td>${r.count}</td>
                <td>${r.avg_magnitude}</td>
                <td>${r.max_magnitude}</td>
                <td>${r.avg_depth}</td>
            </tr>
        `).join("");

    } catch (err) {
        console.error("Dashboard load failed:", err);
    }
}

function renderChart(canvasId, type, data, extraOpts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return; // Skip if canvas doesn't exist
    if (charts[canvasId]) charts[canvasId].destroy();
    const ctx = canvas.getContext("2d");
    charts[canvasId] = new Chart(ctx, {
        type,
        data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: "#1a1a1a" }, beginAtZero: true },
            },
            ...extraOpts,
        },
    });
}

// ─── Map ─────────────────────────────────────────────────────
const mapState = { initialized: false, map: null, markers: null };

async function initMap() {
    try {
        mapState.map = L.map("map", {
            zoomControl: true,
        }).setView([22.5, 80], 4);

        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '© <a href="https://www.openstreetmap.org/">OSM</a> © <a href="https://carto.com/">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 18,
        }).addTo(mapState.map);

        mapState.markers = L.layerGroup().addTo(mapState.map);
        await loadMapData();

        // Populate region filter
        const regions = await apiFetch("/regions");
        const select = document.getElementById("map-region-filter");
        regions.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r;
            opt.textContent = r;
            select.appendChild(opt);
        });

        select.addEventListener("change", () => loadMapData(select.value));

        mapState.initialized = true;
    } catch (err) {
        console.error("Map init failed:", err);
    }
}

async function loadMapData(region = "") {
    try {
        let url = "/data?limit=500";
        if (region) url += `&region=${encodeURIComponent(region)}`;
        const data = await apiFetch(url);

        mapState.markers.clearLayers();

        data.forEach(q => {
            if (!q.latitude || !q.longitude) return; // Skip invalid coords

            let color = "#22c55e";
            if (q.magnitude >= 3) color = "#3b82f6";
            if (q.magnitude >= 4) color = "#f59e0b";
            if (q.magnitude >= 5) color = "#ef4444";

            const radius = Math.max(q.magnitude * 1.8, 2);

            L.circleMarker([q.latitude, q.longitude], {
                radius,
                fillColor: color,
                color: color,
                weight: 0.5,
                fillOpacity: 0.75,
            }).addTo(mapState.markers)
              .bindPopup(`
                <div style="line-height:1.6">
                  <b style="font-size:14px">${q.magnitude}</b> magnitude<br>
                  <span style="color:#b3b3b3">Region:</span> ${q.region}<br>
                  <span style="color:#b3b3b3">Depth:</span> ${q.depth} km<br>
                  <span style="color:#b3b3b3">Date:</span> ${(q.origin_time || "").split("T")[0]}
                </div>
              `);
        });
    } catch (err) {
        console.error("Map data load failed:", err);
    }
}

// ─── Prediction ──────────────────────────────────────────────
async function loadRegionDropdowns() {
    try {
        const regions = await apiFetch("/regions");
        const select = document.getElementById("p-region");
        regions.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r;
            opt.textContent = r;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Region dropdown failed:", err);
    }
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = document.getElementById("predict-btn");
    const spinner = document.getElementById("predict-spinner");
    const btnText = btn.querySelector(".btn-text");

    btnText.textContent = "Predicting...";
    spinner.classList.remove("hidden");

    const payload = {
        region: document.getElementById("p-region").value,
        depth: parseFloat(document.getElementById("p-depth").value),
        year: parseInt(document.getElementById("p-year").value),
        month: parseInt(document.getElementById("p-month").value),
        day: parseInt(document.getElementById("p-day").value),
        hour: parseInt(document.getElementById("p-hour").value),
        dayofweek: new Date(
            document.getElementById("p-year").value,
            document.getElementById("p-month").value - 1,
            document.getElementById("p-day").value
        ).getDay(),
    };

    try {
        const res = await fetch(`${API}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const result = await res.json();

        const magEl = document.getElementById("result-magnitude");
        const regionEl = document.getElementById("result-region");
        const confEl = document.getElementById("result-confidence");

        if (result.error) {
            magEl.textContent = "ERR";
            magEl.className = "result-value high";
            regionEl.textContent = result.error;
            confEl.textContent = "";
            confEl.className = "result-confidence";
        } else {
            const mag = result.magnitude.toFixed(2);
            magEl.textContent = mag;
            magEl.className = "result-value " + (mag < 3 ? "low" : mag < 4.5 ? "mid" : "high");
            regionEl.textContent = `Region: ${result.region}`;
            confEl.textContent = result.confidence;
            confEl.className = `result-confidence ${result.confidence}`;
        }
    } catch (err) {
        console.error("Prediction failed:", err);
        document.getElementById("result-magnitude").textContent = "—";
    } finally {
        btnText.textContent = "Run Prediction";
        spinner.classList.add("hidden");
    }
});

// ─── Analytics ───────────────────────────────────────────────
let analyticsLoaded = false;

async function loadAnalytics() {
    try {
        const [timeData, anomalyData, allData] = await Promise.all([
            apiFetch("/analytics/time"),
            apiFetch("/analytics/anomaly"),
            apiFetch("/data?limit=2000"),
        ]);

        // Time series line chart
        renderChart("timeSeriesChart", "line", {
            labels: timeData.map(d => d.time),
            datasets: [{
                label: "Earthquake Count",
                data: timeData.map(d => d.count),
                borderColor: "#3b82f6",
                backgroundColor: "rgba(59,130,246,0.08)",
                borderWidth: 1.5,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHitRadius: 10,
            }]
        });

        // Scatter: depth vs magnitude
        if (charts["scatterChart"]) charts["scatterChart"].destroy();
        const scatterCtx = document.getElementById("scatterChart").getContext("2d");
        charts["scatterChart"] = new Chart(scatterCtx, {
            type: "scatter",
            data: {
                datasets: [{
                    label: "Depth vs Magnitude",
                    data: allData.slice(0, 800).map(d => ({ x: d.depth, y: d.magnitude })),
                    backgroundColor: "rgba(59,130,246,0.35)",
                    borderColor: "#3b82f6",
                    borderWidth: 0.5,
                    pointRadius: 2.5,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        title: { display: true, text: "Depth (km)", color: "#737373" },
                        grid: { color: "#1a1a1a" },
                    },
                    y: {
                        title: { display: true, text: "Magnitude", color: "#737373" },
                        grid: { color: "#1a1a1a" },
                    },
                },
            },
        });

        // Anomaly table
        const tbody = document.getElementById("anomaly-table-body");
        tbody.innerHTML = anomalyData.map(a => `
            <tr>
                <td>${a.region}</td>
                <td style="color:${a.magnitude >= 5 ? '#ef4444' : '#f59e0b'};font-weight:600">${a.magnitude}</td>
                <td>${a.depth}</td>
                <td>${a.z_score}</td>
                <td>${(a.origin_time || "").split("T")[0]}</td>
            </tr>
        `).join("");

        analyticsLoaded = true;
    } catch (err) {
        console.error("Analytics load failed:", err);
    }
}

// ─── Health Check & Startup ──────────────────────────────────
async function checkHealth() {
    try {
        const data = await apiFetch("/health");
        if (data.status === "ok") {
            document.querySelector(".status-dot").classList.add("online");
            document.getElementById("status-text").textContent = "API Online";
        }
    } catch {
        document.getElementById("status-text").textContent = "API Offline";
    }
}

async function init() {
    await checkHealth();
    await loadRegionDropdowns();
    await loadDashboard();

    // Hide loading overlay
    const overlay = document.getElementById("loading-overlay");
    overlay.classList.add("fade-out");
    setTimeout(() => overlay.remove(), 500);
}

document.addEventListener("DOMContentLoaded", init);
