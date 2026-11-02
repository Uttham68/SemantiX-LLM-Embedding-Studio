/* =========================================================================
   SemantiX LLM Embedding Studio — Pure Pitch-Black OLED & Crisp Emerald
   Zero Warmth • Spacious Symmetrical Grid • 60fps Cursor Graphics
   ========================================================================= */

const PALETTE = [
  "#10B981", // Electric Emerald
  "#14B8A6", // Crisp Teal-Mint
  "#EAB308", // Solar Gold
  "#F97316", // Vibrant Flame Coral
  "#F43F5E", // Crisp Crimson Rose
  "#84CC16", // Vivid Lime
  "#34D399", // Luminous Mint
  "#FACC15", // Bright Gold
  "#FB923C", // Bright Coral
  "#E4E4E7"  // Crisp Platinum
];

/* Crisp High-Contrast Domain Semantic Tokens (Zero Warm Mud / Zero Blue-Purple) */
const CATEGORY_COLORS = {
  "Technology": "#10B981", // Electric Emerald
  "Healthcare": "#14B8A6", // Crisp Teal-Mint
  "Finance": "#EAB308",    // Solar Gold
  "Sports": "#F97316",     // Vibrant Flame Coral
  "Politics": "#F43F5E"    // Crisp Crimson Rose
};

const RANDOM_DOMAIN_POOL = {
  Sports: [
    "India defeated Australia in the final cricket championship match after a century by the opening batsman.",
    "The striker scored a decisive stoppage-time goal to win the European football league trophy.",
    "Grand slam tennis finalists competed in a thrilling five-set match on center court at Wimbledon.",
    "Olympic track and field sprinters broke the world record during the athletics championship finals.",
    "The national basketball team won the playoff championship finals after a dramatic fourth-quarter rally.",
    "Fast bowlers delivered yorkers during the decisive T20 international cricket innings."
  ],
  Technology: [
    "Quantum computers and neural networks are advancing rapidly in Silicon Valley as labs scale qubits.",
    "NASA launched a new space telescope to study distant exoplanets and spiral galaxies.",
    "Zero-trust cybersecurity encryption protects cloud microservices against ransomware and phishing attacks.",
    "Autonomous robotics systems equipped with computer vision sensors automate warehouse manufacturing.",
    "Large language models and transformer neural architectures accelerate natural language processing.",
    "Serverless Kubernetes containers scale distributed cloud applications across global data centers."
  ],
  Finance: [
    "The stock market recorded a significant increase as Wall Street equities rallied on strong earnings.",
    "The central bank hiked benchmark interest rates to control retail inflation and stabilize bond yields.",
    "Institutional investors rebalanced fixed-income portfolios following strong quarterly corporate dividends.",
    "Venture capital underwriting accelerated as fintech startups prepared for initial public offerings.",
    "Global trade balances and gross domestic product growth improved on strong manufacturing exports.",
    "Commercial banks reported higher mortgage lending stability and digital payment revenue."
  ],
  Politics: [
    "Parliamentary lawmakers debated constitutional legislation and national budget policy in the assembly.",
    "Diplomatic ambassadors negotiated a multilateral international peacekeeping and trade treaty.",
    "National election campaigns focused on voter turnout, coalition alliances, and constitutional reform.",
    "Senators ratified the bipartisan infrastructure appropriations bill after committee hearings.",
    "Supreme court justices reviewed regulatory oversight mandates and constitutional rights petitions.",
    "Cabinet ministers announced public fiscal governance reforms during the parliamentary session."
  ],
  Healthcare: [
    "Machine learning and medical imaging algorithms are improving clinical diagnosis in hospitals.",
    "Clinical trials demonstrated strong mRNA vaccine efficacy and antibody response against viral pathogens.",
    "Oncologists prescribed targeted immunotherapy following biopsy screening and genomic sequencing.",
    "Cardiovascular surgeons monitored postoperative intensive care vitals to improve patient recovery.",
    "Public health epidemiologists tracked infectious disease transmission and preventive vaccination.",
    "Radiology screening and biomarker detection enable early intervention for chronic metabolic disorders."
  ]
};

const SMART_SIMILARITY_TRIPLETS = [
  {
    related1: "Artificial intelligence is transforming healthcare.",
    related2: "Machine learning is improving medical diagnosis.",
    unrelated: "India won the cricket championship."
  },
  {
    related1: "Neural network algorithms have drastically boosted machine comprehension of human speech and text.",
    related2: "The development of deep learning architectures has significantly improved automated natural language understanding.",
    unrelated: "The government announced new fiscal stimulus measures to control rising retail inflation."
  },
  {
    related1: "The stock market recorded a significant increase.",
    related2: "Equity indices surged as corporate earnings beat expectations.",
    unrelated: "The striker scored a decisive goal in the football championship final."
  },
  {
    related1: "Zero-trust encryption shields enterprise cloud servers from ransomware exploits.",
    related2: "Cybersecurity firewalls and cryptographic protocols prevent unauthorized network breaches.",
    unrelated: "Olympic marathon runners paced themselves through the final stadium lap."
  },
  {
    related1: "Randomized clinical trials confirmed the efficacy of the new mRNA therapeutic vaccine.",
    related2: "Hospital patient cohorts showed strong immune protection after receiving the antiviral immunization.",
    unrelated: "NASA launched an orbital space telescope to observe distant exoplanets."
  },
  {
    related1: "Parliamentary legislators ratified the bipartisan constitutional reform bill.",
    related2: "Lawmakers in the national assembly voted to approve the new statutory legislation.",
    unrelated: "Semiconductor foundries unveiled next-generation GPU accelerators for robotics."
  }
];

let lastRandomText = "";

function pickRandomSentence(excludeText = "") {
  const allSentences = Object.values(RANDOM_DOMAIN_POOL).flat();
  const candidates = allSentences.filter((s) => s !== excludeText && s !== lastRandomText);
  const picked = candidates[Math.floor(Math.random() * candidates.length)];
  lastRandomText = picked;
  return picked;
}

let galaxyState = {
  points: [],
  prevCoords: new Map(),
  morphProgress: 1,
  morphAnimId: null,
  pulsePhase: 0,
  pulseAnimId: null,
  screenPoints: [],
  activeFilter: null,
  colorMode: "cluster",
  liveQuery: null
};

let currentSimPairs = [];
let simAnimId = null;
let simPulsePhase = 0;

document.addEventListener("DOMContentLoaded", () => {
  initCursorIlluminationCanvas();
  initThemeToggle();
  initSpotlightCards();
  initTabs();
  initClassificationTab();
  initSimilarityTab();
  initClusteringTab();
  initTopicsTab();
  initGalleryModal();

  // Initial data fetches
  loadOverviewAndBenchmarks();
  runClassification();
  runSimilarity();
  loadClusteringData();
  loadTopicsData();
  runCustomBatch();
});

/* =========================================================================
   0. INTERACTIVE CURSOR-ILLUMINATED ARCHITECTURAL GRAPHICS ENGINE
   ========================================================================= */
function initCursorIlluminationCanvas() {
  const canvas = document.getElementById("cursor-illumination-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W = (canvas.width = window.innerWidth);
  let H = (canvas.height = window.innerHeight);

  const mouse = {
    x: W * 0.5,
    y: H * 0.3,
    targetX: W * 0.5,
    targetY: H * 0.3,
    active: false
  };

  window.addEventListener("resize", () => {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  });

  window.addEventListener("mousemove", (e) => {
    mouse.targetX = e.clientX;
    mouse.targetY = e.clientY;
    mouse.active = true;
  });

  // Create floating architectural vector nodes
  const NODE_COUNT = 38;
  const nodes = Array.from({ length: NODE_COUNT }, (_, i) => ({
    x: Math.random() * W,
    y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.45,
    vy: (Math.random() - 0.5) * 0.45,
    r: 1.6 + (i % 3) * 0.7,
    isGold: i % 2 === 0
  }));

  function renderFrame() {
    // Smooth cursor interpolation
    mouse.x += (mouse.targetX - mouse.x) * 0.14;
    mouse.y += (mouse.targetY - mouse.y) * 0.14;

    ctx.clearRect(0, 0, W, H);
    const isLight = document.documentElement.getAttribute("data-theme") === "light";

    // 1. Ambient Cursor Torchlight Glow (Crisp Emerald & Neutral Platinum on #000000)
    const torchRadius = 420;
    const grad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, torchRadius);
    if (isLight) {
      grad.addColorStop(0, "rgba(16, 185, 129, 0.08)");
      grad.addColorStop(0.5, "rgba(20, 184, 166, 0.03)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");
    } else {
      grad.addColorStop(0, "rgba(16, 185, 129, 0.13)");
      grad.addColorStop(0.42, "rgba(255, 255, 255, 0.035)");
      grad.addColorStop(1, "rgba(0, 0, 0, 0)");
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // 2. Precision Architectural Crosshairs (+) that illuminate & react near cursor
    const step = 64;
    const illuminateRadius = 280;
    ctx.lineWidth = 1;

    for (let gx = step / 2; gx < W; gx += step) {
      for (let gy = step / 2; gy < H; gy += step) {
        const dx = gx - mouse.x;
        const dy = gy - mouse.y;
        const dist = Math.hypot(dx, dy);
        if (dist < illuminateRadius) {
          const intensity = Math.pow(1 - dist / illuminateRadius, 1.75);
          // Subtle magnetic ripple push away from cursor
          const push = intensity * 4.5;
          const angle = Math.atan2(dy, dx);
          const rx = gx + Math.cos(angle) * push;
          const ry = gy + Math.sin(angle) * push;
          const arm = 3 + intensity * 4;
          ctx.strokeStyle = isLight
            ? `rgba(16, 185, 129, ${0.14 + intensity * 0.55})`
            : `rgba(52, 211, 153, ${0.10 + intensity * 0.65})`;
          ctx.beginPath();
          ctx.moveTo(rx - arm, ry);
          ctx.lineTo(rx + arm, ry);
          ctx.moveTo(rx, ry - arm);
          ctx.lineTo(rx, ry + arm);
          ctx.stroke();
        }
      }
    }

    // 3. Floating Vector Nodes & Cursor-Illuminated Geometric Filaments
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;

      const distToMouse = Math.hypot(n.x - mouse.x, n.y - mouse.y);
      const nearMouse = distToMouse < 260;
      const mouseFactor = nearMouse ? 1 - distToMouse / 260 : 0;

      // Connect node to cursor with glowing emerald / platinum filament
      if (nearMouse) {
        ctx.beginPath();
        ctx.moveTo(n.x, n.y);
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = n.isGold
          ? `rgba(52, 211, 153, ${mouseFactor * 0.40})`
          : `rgba(255, 255, 255, ${mouseFactor * 0.28})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Connect nearby nodes if illuminated by cursor
      for (let j = i + 1; j < nodes.length; j++) {
        const n2 = nodes[j];
        const dNodes = Math.hypot(n.x - n2.x, n.y - n2.y);
        if (dNodes < 140) {
          const pairBoost = Math.max(mouseFactor, 0.12);
          const alpha = (1 - dNodes / 140) * pairBoost * 0.38;
          ctx.beginPath();
          ctx.moveTo(n.x, n.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
          ctx.lineWidth = 0.85;
          ctx.stroke();
        }
      }

      // Draw node point
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + mouseFactor * 2.0, 0, Math.PI * 2);
      ctx.fillStyle = n.isGold
        ? `rgba(52, 211, 153, ${0.22 + mouseFactor * 0.75})`
        : `rgba(255, 255, 255, ${0.18 + mouseFactor * 0.70})`;
      ctx.fill();
    }

    requestAnimationFrame(renderFrame);
  }

  requestAnimationFrame(renderFrame);
}

/* =========================================================================
   0b. UX HELPERS: TOASTS, SMOOTH COUNTERS, SPOTLIGHT & THEME
   ========================================================================= */
function showToast(message, icon = "✨") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const item = document.createElement("div");
  item.className = "toast-item";
  item.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  container.appendChild(item);

  setTimeout(() => {
    item.style.transition = "opacity 0.25s ease, transform 0.25s ease";
    item.style.opacity = "0";
    item.style.transform = "translateY(8px)";
    setTimeout(() => item.remove(), 260);
  }, 2400);
}

function flashInput(el) {
  if (!el) return;
  el.classList.remove("input-flash");
  void el.offsetWidth;
  el.classList.add("input-flash");
}

function updateCharTokenCounter() {
  const inputEl = document.getElementById("classify-input");
  const badgeEl = document.getElementById("char-token-counter");
  if (!inputEl || !badgeEl) return;
  const text = inputEl.value.trim();
  const chars = text.length;
  const words = text ? text.split(/\s+/).length : 0;
  const estTokens = Math.max(0, Math.round(words * 1.3));
  badgeEl.textContent = `${chars} chars • ~${estTokens} tokens`;
}

function animateNumber(el, targetVal, decimals = 1, suffix = "", prefix = "", duration = 420) {
  if (!el) return;
  const currentText = el.textContent.replace(/[^0-9.-]/g, "");
  const startVal = parseFloat(currentText);
  if (isNaN(startVal) || Math.abs(startVal - targetVal) < 1e-6) {
    el.textContent = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
    return;
  }

  const startTime = performance.now();
  function step(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 4);
    const val = startVal + (targetVal - startVal) * eased;
    el.textContent = `${prefix}${val.toFixed(decimals)}${suffix}`;
    if (t < 1) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

function initSpotlightCards() {
  document.addEventListener("mousemove", (e) => {
    document.querySelectorAll(".spotlight-card").forEach((card) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);

      // Illuminate border when cursor is within 140px of card bounds
      const dx = Math.max(rect.left - e.clientX, 0, e.clientX - rect.right);
      const dy = Math.max(rect.top - e.clientY, 0, e.clientY - rect.bottom);
      if (Math.hypot(dx, dy) < 140) {
        card.classList.add("near-cursor");
      } else {
        card.classList.remove("near-cursor");
      }
    });
  });
}

function initThemeToggle() {
  const btn = document.getElementById("theme-toggle-btn");
  const iconEl = document.getElementById("theme-icon");
  const labelEl = document.getElementById("theme-label");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const html = document.documentElement;
    const isLight = html.getAttribute("data-theme") === "light";
    const nextTheme = isLight ? "dark" : "light";
    html.setAttribute("data-theme", nextTheme);

    if (nextTheme === "light") {
      iconEl.textContent = "☀️";
      labelEl.textContent = "Daylight";
      showToast("Switched to Crisp Daylight Theme", "☀️");
    } else {
      iconEl.textContent = "✨";
      labelEl.textContent = "Pitch Black";
      showToast("Switched to Pure Pitch-Black OLED Mode", "✨");
    }

    drawSimilarityTriangle(currentSimPairs);
    drawGalaxyCanvas();
  });
}

/* =========================================================================
   1. TAB NAVIGATION & SLIDING GLIDER PILL
   ========================================================================= */
function updateTabGlider() {
  const activeBtn = document.querySelector(".tab-btn.active");
  const glider = document.getElementById("tab-glider");
  if (!activeBtn || !glider) return;
  glider.style.width = `${activeBtn.offsetWidth}px`;
  glider.style.transform = `translateX(${activeBtn.offsetLeft - 8}px)`;
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  function activateTab(btn) {
    buttons.forEach((b) => b.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add("active");

    updateTabGlider();

    if (btn.dataset.tab === "tab-clustering") {
      startGalaxyPulseLoop();
    } else {
      stopGalaxyPulseLoop();
    }

    if (btn.dataset.tab === "tab-similarity") {
      startSimilarityTriangleLoop();
    } else {
      stopSimilarityTriangleLoop();
    }
  }

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn));
  });

  window.addEventListener("resize", updateTabGlider);
  setTimeout(updateTabGlider, 60);

  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key >= "1" && e.key <= "5") {
      const idx = parseInt(e.key, 10) - 1;
      if (buttons[idx]) activateTab(buttons[idx]);
    }
  });
}

/* =========================================================================
   2. OVERVIEW KPI & BENCHMARK TABLE (WITH ★ BEST OVERALL HIGHLIGHT)
   ========================================================================= */
async function loadOverviewAndBenchmarks() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    document.getElementById("kpi-docs").textContent = `${data.total_documents.toLocaleString()} Docs`;
    animateNumber(document.getElementById("kpi-acc"), data.test_accuracy * 100, 1, "%");
    document.getElementById("kpi-f1").textContent = `Macro F1: ${data.test_f1.toFixed(4)}`;
    animateNumber(document.getElementById("kpi-nmi"), data.clustering_nmi, 4);
    document.getElementById("kpi-ari").textContent = `ARI: ${data.clustering_ari.toFixed(4)} (K=5)`;
    animateNumber(document.getElementById("kpi-div"), data.topic_diversity * 100, 1, "%");

    const tbody = document.querySelector("#benchmark-model-table tbody");
    tbody.innerHTML = "";
    data.model_comparison.forEach((row, idx) => {
      const modelName = String(row["Representation Model"]);
      const isWinner = modelName.includes("all-MiniLM-L6-v2");
      const tr = document.createElement("tr");
      tr.style.setProperty("--i", idx);
      if (isWinner) tr.className = "best-overall-row";

      const winnerBadge = isWinner
        ? `<span class="best-overall-badge">★ Best Overall</span>`
        : "";

      tr.innerHTML = `
        <td><strong>${modelName}</strong>${winnerBadge}</td>
        <td class="num-col"><code>${row["Vector Dim"]}D</code></td>
        <td class="num-col"><strong>${(row["Test Accuracy"] * 100).toFixed(1)}%</strong></td>
        <td class="num-col"><strong>${row["Test Macro F1"].toFixed(4)}</strong></td>
        <td class="num-col"><span class="verdict-pill ${row["Paraphrase Generalization Acc"] > 0.8 ? 'verdict-high' : 'verdict-medium'}">${(row["Paraphrase Generalization Acc"] * 100).toFixed(1)}%</span></td>
        <td class="num-col"><code>${row["Clustering Silhouette"].toFixed(4)}</code></td>
        <td class="num-col"><code>${row["Clustering ARI"].toFixed(4)}</code></td>
        <td class="num-col"><strong>${row["Clustering NMI"].toFixed(4)}</strong></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

/* =========================================================================
   3. TAB 1: LIVE CLASSIFICATION & COSINE NEAREST NEIGHBORS
   ========================================================================= */
function initClassificationTab() {
  const inputEl = document.getElementById("classify-input");
  if (inputEl) {
    inputEl.addEventListener("input", updateCharTokenCounter);
    updateCharTokenCounter();
  }

  document.querySelectorAll("#tab-classify .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      inputEl.value = chip.dataset.sample;
      updateCharTokenCounter();
      flashInput(inputEl);
      runClassification();
    });
  });

  document.getElementById("btn-run-classify").addEventListener("click", runClassification);
  document.getElementById("classifier-select").addEventListener("change", runClassification);

  const randBtn = document.getElementById("btn-random-classify");
  if (randBtn) {
    randBtn.addEventListener("click", () => {
      inputEl.value = pickRandomSentence(inputEl.value);
      updateCharTokenCounter();
      flashInput(inputEl);
      runClassification();
    });
  }

  const jumpBtn = document.getElementById("btn-jump-galaxy");
  if (jumpBtn) {
    jumpBtn.addEventListener("click", () => {
      const tabBtn = document.querySelector('.tab-btn[data-tab="tab-clustering"]');
      if (tabBtn) {
        tabBtn.click();
        showToast("Plotted your live query star (⭐) in the 2D Semantic Galaxy!", "🌌");
      }
    });
  }
}

async function runClassification() {
  const text = document.getElementById("classify-input").value.trim();
  const clf = document.getElementById("classifier-select").value;
  if (!text) return;

  const spinner = document.getElementById("classify-spinner");
  const telemetryCard = document.getElementById("classify-telemetry-card");
  if (spinner) spinner.classList.remove("hidden");
  if (telemetryCard) telemetryCard.classList.add("skeleton-loading");

  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, classifier_type: clf })
    });
    const data = await res.json();
    if (data.error) return;

    const domain = data.predicted_category;
    const domainColor = CATEGORY_COLORS[domain] || "#10B981";

    const heroBox = document.getElementById("pred-hero-box");
    if (heroBox) heroBox.setAttribute("data-domain", domain);

    const dotEl = document.getElementById("pred-domain-dot");
    if (dotEl) {
      dotEl.style.background = domainColor;
      dotEl.style.color = domainColor;
    }

    const catEl = document.getElementById("pred-category-text");
    catEl.textContent = domain;
    animateNumber(document.getElementById("pred-conf-text"), data.confidence_pct, 1, "%");
    animateNumber(document.getElementById("pred-time-text"), data.encoding_time_ms, 1, " ms", "⚡ ");

    const statusBadge = document.getElementById("pred-status-badge");
    const confEl = document.getElementById("pred-conf-text");
    if (data.is_low_confidence) {
      statusBadge.textContent = "⚠️ LOW CONFIDENCE / BORDERLINE DOMAIN";
      statusBadge.style.color = "#EAB308";
      confEl.style.color = "#EAB308";
    } else {
      statusBadge.textContent = "PREDICTED DOMAIN CATEGORY";
      statusBadge.style.color = domainColor;
      confEl.style.color = "#34D399";
    }

    // Render 650ms animated probability bars with Domain Color Tokens
    const barsContainer = document.getElementById("prob-bars");
    barsContainer.innerHTML = "";
    data.probabilities.forEach((item) => {
      const catColor = CATEGORY_COLORS[item.Category] || "#10B981";
      const row = document.createElement("div");
      row.className = "prob-row";
      const fillId = `prob-fill-${item.Category}`;
      row.innerHTML = `
        <div class="prob-header">
          <span style="display:inline-flex; align-items:center; gap:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:${catColor};"></span>
            ${item.Category}
          </span>
          <span class="prob-pct-mono">${item.Percentage.toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" id="${fillId}" style="width: 0%; background: linear-gradient(90deg, ${catColor}, ${catColor}cc)"></div>
        </div>
      `;
      barsContainer.appendChild(row);
      requestAnimationFrame(() => {
        const fillEl = document.getElementById(fillId);
        if (fillEl) fillEl.style.width = `${Math.max(2, item.Percentage)}%`;
      });
    });

    // Render Top-4 Nearest Corpus Neighbors with cosine meter bar + domain pill + hover expand
    const grid = document.getElementById("nearest-docs-grid");
    grid.innerHTML = "";
    data.nearest_documents.forEach((doc, idx) => {
      const card = document.createElement("div");
      card.className = "neighbor-card";
      card.style.setProperty("--i", idx);
      const meterId = `cos-meter-${idx}`;
      const pctSim = Math.min(100, Math.max(8, Math.round(doc.cosine_similarity * 100)));
      card.innerHTML = `
        <div class="neighbor-top">
          <span class="domain-pill" data-domain="${doc.category}">${doc.category} • ${doc.subtopic}</span>
          <span class="sim-score-pill">cos = ${doc.cosine_similarity.toFixed(3)}</span>
        </div>
        <div class="cos-meter-track">
          <div class="cos-meter-fill" id="${meterId}" style="width: 0%;"></div>
        </div>
        <div class="neighbor-text" title="Hover to expand full document text">${doc.text}</div>
      `;
      grid.appendChild(card);
      requestAnimationFrame(() => {
        const mEl = document.getElementById(meterId);
        if (mEl) mEl.style.width = `${pctSim}%`;
      });
    });

    // Save 2D query star projection for Tab 3 Semantic Galaxy
    if (data.query_2d) {
      galaxyState.liveQuery = {
        text,
        predicted_category: data.predicted_category,
        confidence_pct: data.confidence_pct,
        tsne: data.query_2d.tsne,
        pca: data.query_2d.pca,
        neighbors: data.query_2d.neighbors || []
      };
      const gInput = document.getElementById("galaxy-query-input");
      if (gInput) gInput.value = text;
      drawGalaxyCanvas();
    }
  } catch (err) {
    console.error("Classification error:", err);
  } finally {
    if (spinner) spinner.classList.add("hidden");
    if (telemetryCard) telemetryCard.classList.remove("skeleton-loading");
  }
}

/* =========================================================================
   4. TAB 2: SEMANTIC VS. LEXICAL LAB (DUAL GAUGES + SEMANTIC GAIN Δ)
   ========================================================================= */
function initSimilarityTab() {
  document.querySelectorAll(".sim-pack").forEach((btn) => {
    btn.addEventListener("click", () => {
      const elA = document.getElementById("sim-a");
      const elB = document.getElementById("sim-b");
      const elC = document.getElementById("sim-c");
      elA.value = btn.dataset.a;
      elB.value = btn.dataset.b;
      elC.value = btn.dataset.c;
      flashInput(elA);
      flashInput(elB);
      flashInput(elC);
      runSimilarity();
    });
  });

  document.getElementById("btn-run-similarity").addEventListener("click", runSimilarity);

  const btnRandA = document.getElementById("btn-rand-a");
  const btnRandB = document.getElementById("btn-rand-b");
  const btnRandC = document.getElementById("btn-rand-c");
  const btnRandTriplet = document.getElementById("btn-rand-triplet");

  if (btnRandA) {
    btnRandA.addEventListener("click", () => {
      const el = document.getElementById("sim-a");
      el.value = pickRandomSentence(el.value);
      flashInput(el);
      runSimilarity();
    });
  }
  if (btnRandB) {
    btnRandB.addEventListener("click", () => {
      const el = document.getElementById("sim-b");
      el.value = pickRandomSentence(el.value);
      flashInput(el);
      runSimilarity();
    });
  }
  if (btnRandC) {
    btnRandC.addEventListener("click", () => {
      const el = document.getElementById("sim-c");
      el.value = pickRandomSentence(el.value);
      flashInput(el);
      runSimilarity();
    });
  }
  if (btnRandTriplet) {
    btnRandTriplet.addEventListener("click", () => {
      const t = SMART_SIMILARITY_TRIPLETS[Math.floor(Math.random() * SMART_SIMILARITY_TRIPLETS.length)];
      const arr = [t.related1, t.related2, t.unrelated];
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      const elA = document.getElementById("sim-a");
      const elB = document.getElementById("sim-b");
      const elC = document.getElementById("sim-c");
      elA.value = arr[0];
      elB.value = arr[1];
      elC.value = arr[2];
      flashInput(elA);
      flashInput(elB);
      flashInput(elC);
      runSimilarity();
    });
  }
}

async function runSimilarity() {
  const text_a = document.getElementById("sim-a").value.trim();
  const text_b = document.getElementById("sim-b").value.trim();
  const text_c = document.getElementById("sim-c").value.trim();
  if (!text_a || !text_b || !text_c) return;

  const spinner = document.getElementById("sim-spinner");
  if (spinner) spinner.classList.remove("hidden");

  try {
    const res = await fetch("/api/similarity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text_a, text_b, text_c })
    });
    const data = await res.json();
    if (data.error) return;

    const best = data.best_pair;
    const bestGain = best.llm_sim - best.tfidf_sim;
    const banner = document.getElementById("sim-best-banner");
    banner.innerHTML = `
      <strong>🏆 Most Semantically Related Pair:</strong> <code>${best.pair}</code>
      — LLM Cosine = <strong>${best.llm_sim.toFixed(4)}</strong> vs. TF-IDF = <code>${best.tfidf_sim.toFixed(4)}</code>
      <span class="delta-gain-badge" style="margin-left:6px;">+${bestGain.toFixed(4)} Semantic Gain</span>
    `;

    // Render Side-by-Side Dual Progress Gauges
    const gaugesEl = document.getElementById("sim-dual-gauges");
    if (gaugesEl) {
      gaugesEl.innerHTML = "";
      data.pairs.forEach((p, idx) => {
        const delta = p.llm_sim - p.tfidf_sim;
        const deltaSign = delta >= 0 ? "+" : "";
        const badgeClass =
          p.verdict_level === "high"
            ? "verdict-high"
            : p.verdict_level === "medium"
            ? "verdict-medium"
            : "verdict-low";
        const icon = p.verdict_level === "high" ? "🟢" : p.verdict_level === "medium" ? "🟡" : "🔴";

        const card = document.createElement("div");
        card.className = "sim-gauge-card";
        card.style.setProperty("--i", idx);
        const tfidfBarId = `gauge-tfidf-${idx}`;
        const llmBarId = `gauge-llm-${idx}`;

        card.innerHTML = `
          <div class="sim-gauge-header">
            <strong>${p.pair}</strong>
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="delta-gain-badge">${deltaSign}${delta.toFixed(4)} Semantic Gain</span>
              <span class="verdict-pill ${badgeClass}">${icon} ${p.verdict_label.split("(")[0].trim()}</span>
            </div>
          </div>
          <div class="dual-gauge-row">
            <span class="gauge-label">TF-IDF Lexical</span>
            <div class="prob-track">
              <div class="prob-fill" id="${tfidfBarId}" style="width:0%; background:#71717A;"></div>
            </div>
            <span class="gauge-val-mono">${p.tfidf_sim.toFixed(4)}</span>
          </div>
          <div class="dual-gauge-row">
            <span class="gauge-label" style="color:#34D399;">LLM 384D Cosine</span>
            <div class="prob-track">
              <div class="prob-fill" id="${llmBarId}" style="width:0%; background:linear-gradient(90deg, #10B981, #34D399);"></div>
            </div>
            <span class="gauge-val-mono" style="color:#34D399;">${p.llm_sim.toFixed(4)}</span>
          </div>
        `;
        gaugesEl.appendChild(card);

        requestAnimationFrame(() => {
          const tEl = document.getElementById(tfidfBarId);
          const lEl = document.getElementById(llmBarId);
          if (tEl) tEl.style.width = `${Math.max(2, Math.min(100, p.tfidf_sim * 100))}%`;
          if (lEl) lEl.style.width = `${Math.max(2, Math.min(100, Math.max(0, p.llm_sim) * 100))}%`;
        });
      });
    }

    // Render Comparison Table
    const tbody = document.querySelector("#sim-results-table tbody");
    tbody.innerHTML = "";
    data.pairs.forEach((p, idx) => {
      const badgeClass =
        p.verdict_level === "high"
          ? "verdict-high"
          : p.verdict_level === "medium"
          ? "verdict-medium"
          : "verdict-low";
      const icon = p.verdict_level === "high" ? "🟢" : p.verdict_level === "medium" ? "🟡" : "🔴";
      const sharedText = p.shared_count === 0 ? "0 (Zero Overlap)" : `${p.shared_count} (${p.shared_words.join(", ")})`;
      const delta = p.llm_sim - p.tfidf_sim;
      const deltaSign = delta >= 0 ? "+" : "";

      const tr = document.createElement("tr");
      tr.style.setProperty("--i", idx);
      tr.innerHTML = `
        <td><strong>${p.pair}</strong></td>
        <td><code>${sharedText}</code></td>
        <td class="num-col"><code>${p.tfidf_sim.toFixed(4)}</code></td>
        <td class="num-col"><strong style="color:#34D399;">${p.llm_sim.toFixed(4)}</strong></td>
        <td class="num-col"><span class="delta-gain-badge">${deltaSign}${delta.toFixed(4)}</span></td>
        <td><span class="verdict-pill ${badgeClass}">${icon} ${p.verdict_label}</span></td>
      `;
      tbody.appendChild(tr);
    });

    currentSimPairs = data.pairs;
    drawSimilarityTriangle(currentSimPairs);
  } catch (err) {
    console.error("Similarity error:", err);
  } finally {
    if (spinner) spinner.classList.add("hidden");
  }
}

function startSimilarityTriangleLoop() {
  if (simAnimId) cancelAnimationFrame(simAnimId);
  function loop() {
    simPulsePhase = (simPulsePhase + 0.014) % 1;
    drawSimilarityTriangle(currentSimPairs);
    simAnimId = requestAnimationFrame(loop);
  }
  simAnimId = requestAnimationFrame(loop);
}

function stopSimilarityTriangleLoop() {
  if (simAnimId) {
    cancelAnimationFrame(simAnimId);
    simAnimId = null;
  }
}

function drawSimilarityTriangle(pairs) {
  const canvas = document.getElementById("sim-triangle-canvas");
  if (!canvas || !pairs || pairs.length < 3) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const nodes = {
    A: { x: W / 2, y: 52, label: "Anchor A", color: "#10B981" },
    B: { x: 108, y: H - 52, label: "Comp. B", color: "#EAB308" },
    C: { x: W - 108, y: H - 52, label: "Comp. C", color: "#F97316" }
  };

  const edges = [
    { from: nodes.A, to: nodes.B, data: pairs[0] },
    { from: nodes.A, to: nodes.C, data: pairs[1] },
    { from: nodes.B, to: nodes.C, data: pairs[2] }
  ];

  edges.forEach((edge) => {
    const sim = edge.data.llm_sim;
    const isHigh = sim >= 0.4;
    const isMed = sim >= 0.22 && sim < 0.4;

    ctx.beginPath();
    ctx.moveTo(edge.from.x, edge.from.y);
    ctx.lineTo(edge.to.x, edge.to.y);
    ctx.lineWidth = isHigh ? 5.5 : isMed ? 3 : 1.6;
    ctx.strokeStyle = isHigh ? "#10B981" : isMed ? "#EAB308" : "rgba(255, 255, 255, 0.22)";
    ctx.stroke();

    if (isHigh) {
      const px = edge.from.x + (edge.to.x - edge.from.x) * simPulsePhase;
      const py = edge.from.y + (edge.to.y - edge.from.y) * simPulsePhase;
      ctx.beginPath();
      ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = "#34D399";
      ctx.shadowColor = "#10B981";
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    const mx = (edge.from.x + edge.to.x) / 2;
    const my = (edge.from.y + edge.to.y) / 2;
    ctx.fillStyle = "#09090B";
    ctx.fillRect(mx - 46, my - 13, 92, 26);
    ctx.strokeStyle = isHigh ? "#10B981" : "rgba(255, 255, 255, 0.22)";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(mx - 46, my - 13, 92, 26);

    ctx.fillStyle = isHigh ? "#34D399" : "#D4D4D8";
    ctx.font = "bold 12px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`cos=${sim.toFixed(3)}`, mx, my);
  });

  Object.values(nodes).forEach((n) => {
    ctx.beginPath();
    ctx.arc(n.x, n.y, 28, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();

    ctx.fillStyle = "#000000";
    ctx.font = "bold 11.5px Plus Jakarta Sans, Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(n.label, n.x, n.y);
  });
}

/* =========================================================================
   5. TAB 3: 2D SEMANTIC GALAXY EXPLORER (WITH 60FPS SMOOTH MORPHING)
   ========================================================================= */
function initClusteringTab() {
  const kSlider = document.getElementById("cluster-k-slider");
  const projSelect = document.getElementById("cluster-proj-select");
  const colorSelect = document.getElementById("cluster-color-mode");

  kSlider.addEventListener("input", () => {
    document.getElementById("k-val-label").textContent = kSlider.value;
  });
  kSlider.addEventListener("change", loadClusteringData);
  projSelect.addEventListener("change", () => {
    loadClusteringData();
    showToast(`Morphing 1,000 vectors to ${projSelect.value.toUpperCase()} manifold...`, "🌌");
  });
  colorSelect.addEventListener("change", () => {
    galaxyState.colorMode = colorSelect.value;
    galaxyState.activeFilter = null;
    renderGalaxyLegend();
    drawGalaxyCanvas();
  });

  const btnGalaxyPlot = document.getElementById("btn-galaxy-plot");
  const btnGalaxyRand = document.getElementById("btn-galaxy-rand");
  const galaxyInput = document.getElementById("galaxy-query-input");

  if (btnGalaxyPlot && galaxyInput) {
    btnGalaxyPlot.addEventListener("click", () => {
      const q = galaxyInput.value.trim();
      if (!q) return;
      document.getElementById("classify-input").value = q;
      updateCharTokenCounter();
      flashInput(galaxyInput);
      runClassification();
      showToast("Updated Live Query Star (⭐) on the 2D Manifold", "⭐");
    });
  }
  if (btnGalaxyRand && galaxyInput) {
    btnGalaxyRand.addEventListener("click", () => {
      const picked = pickRandomSentence(galaxyInput.value);
      galaxyInput.value = picked;
      document.getElementById("classify-input").value = picked;
      updateCharTokenCounter();
      flashInput(galaxyInput);
      runClassification();
    });
  }

  const canvas = document.getElementById("galaxy-canvas");
  const tooltip = document.getElementById("galaxy-tooltip");

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    let hovered = null;
    let minDist = 10;
    for (const sp of galaxyState.screenPoints) {
      const d = Math.hypot(sp.sx - mx, sp.sy - my);
      if (d < minDist) {
        minDist = d;
        hovered = sp.point;
      }
    }

    if (hovered) {
      tooltip.classList.remove("hidden");
      tooltip.style.left = `${Math.min(e.offsetX + 16, rect.width - 305)}px`;
      tooltip.style.top = `${Math.max(10, e.offsetY - 22)}px`;
      tooltip.innerHTML = `
        <div style="font-weight:800; color:#34D399; margin-bottom:4px;">
          ${hovered.doc_id} • ${hovered.true_category} (${hovered.subtopic})
        </div>
        <div style="font-size:11px; color:#A1A1AA; margin-bottom:6px;">
          Assigned: ${hovered.cluster_label}
        </div>
        <div style="color:#FFFFFF; line-height:1.5;">${hovered.snippet}</div>
      `;
    } else {
      tooltip.classList.add("hidden");
    }
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.classList.add("hidden");
  });
}

async function loadClusteringData() {
  const k = document.getElementById("cluster-k-slider").value;
  const proj = document.getElementById("cluster-proj-select").value;

  try {
    const prevMap = new Map();
    if (galaxyState.points.length) {
      const xs = galaxyState.points.map((p) => p.x);
      const ys = galaxyState.points.map((p) => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      galaxyState.points.forEach((p) => {
        const nx = (p.x - minX) / (maxX - minX + 1e-9);
        const ny = (p.y - minY) / (maxY - minY + 1e-9);
        prevMap.set(p.doc_id, { nx, ny });
      });
    }

    const res = await fetch(`/api/clustering?k=${k}&projection=${proj}`);
    const data = await res.json();

    animateNumber(document.getElementById("clust-sil"), data.silhouette_score, 4);
    animateNumber(document.getElementById("clust-ari"), data.ari_score, 4);
    animateNumber(document.getElementById("clust-nmi"), data.nmi_score, 4);
    animateNumber(document.getElementById("clust-inertia"), data.inertia, 2);

    galaxyState.prevCoords = prevMap;
    galaxyState.points = data.points;
    galaxyState.activeFilter = null;

    renderGalaxyLegend();
    animateGalaxyMorph();

    const tbody = document.querySelector("#cluster-summary-table tbody");
    tbody.innerHTML = "";
    data.clusters.forEach((c, idx) => {
      const themeStr = String(c["Emerging Theme"]);
      const matchedDomain = Object.keys(CATEGORY_COLORS).find((d) => themeStr.includes(d)) || "Technology";
      const tr = document.createElement("tr");
      tr.style.setProperty("--i", idx);
      tr.innerHTML = `
        <td><code>${c["Cluster ID"]}</code></td>
        <td><span class="domain-pill" data-domain="${matchedDomain}">${themeStr}</span></td>
        <td class="num-col"><strong>${c["Document Count"]}</strong></td>
        <td class="num-col"><strong>${c["Cluster Purity (%)"].toFixed(1)}%</strong></td>
        <td class="num-col"><code>${c["Avg Centroid Cosine Sim"].toFixed(4)}</code></td>
        <td style="font-size:12.5px; color:var(--text-secondary);">${c["Exemplar Document"]}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Clustering load error:", err);
  }
}

function animateGalaxyMorph() {
  if (!galaxyState.prevCoords.size) {
    galaxyState.morphProgress = 1;
    drawGalaxyCanvas();
    return;
  }
  if (galaxyState.morphAnimId) cancelAnimationFrame(galaxyState.morphAnimId);
  const start = performance.now();
  const duration = 520;

  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    galaxyState.morphProgress = 1 - Math.pow(1 - t, 3);
    drawGalaxyCanvas();
    if (t < 1) {
      galaxyState.morphAnimId = requestAnimationFrame(step);
    }
  }
  galaxyState.morphAnimId = requestAnimationFrame(step);
}

function startGalaxyPulseLoop() {
  if (galaxyState.pulseAnimId) cancelAnimationFrame(galaxyState.pulseAnimId);
  function loop() {
    galaxyState.pulsePhase = (galaxyState.pulsePhase + 0.03) % (Math.PI * 2);
    if (galaxyState.morphProgress >= 1) {
      drawGalaxyCanvas();
    }
    galaxyState.pulseAnimId = requestAnimationFrame(loop);
  }
  galaxyState.pulseAnimId = requestAnimationFrame(loop);
}

function stopGalaxyPulseLoop() {
  if (galaxyState.pulseAnimId) {
    cancelAnimationFrame(galaxyState.pulseAnimId);
    galaxyState.pulseAnimId = null;
  }
}

function renderGalaxyLegend() {
  const legendEl = document.getElementById("galaxy-legend");
  legendEl.innerHTML = "";
  const pts = galaxyState.points;
  if (!pts.length) return;

  const groups = new Map();
  pts.forEach((p) => {
    const key = galaxyState.colorMode === "cluster" ? p.cluster_label : p.true_category;
    const color =
      galaxyState.colorMode === "cluster"
        ? PALETTE[p.cluster_id % PALETTE.length]
        : CATEGORY_COLORS[p.true_category] || "#F59E0B";
    if (!groups.has(key)) {
      groups.set(key, { label: key, color, count: 0 });
    }
    groups.get(key).count += 1;
  });

  groups.forEach((item) => {
    const div = document.createElement("div");
    div.className = `legend-item ${galaxyState.activeFilter === item.label ? "active" : ""}`;
    div.innerHTML = `
      <span class="legend-dot" style="background:${item.color}; color:${item.color}"></span>
      <span style="flex:1;">${item.label}</span>
      <code>${item.count}</code>
    `;
    div.addEventListener("click", () => {
      galaxyState.activeFilter = galaxyState.activeFilter === item.label ? null : item.label;
      renderGalaxyLegend();
      drawGalaxyCanvas();
    });
    legendEl.appendChild(div);
  });
}

function drawGalaxyCanvas() {
  const canvas = document.getElementById("galaxy-canvas");
  if (!canvas || !galaxyState.points.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const pad = 38;

  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(168, 162, 158, 0.06)";
  ctx.lineWidth = 1;
  for (let gx = pad; gx < W - pad; gx += 60) {
    ctx.beginPath();
    ctx.moveTo(gx, pad);
    ctx.lineTo(gx, H - pad);
    ctx.stroke();
  }
  for (let gy = pad; gy < H - pad; gy += 60) {
    ctx.beginPath();
    ctx.moveTo(pad, gy);
    ctx.lineTo(W - pad, gy);
    ctx.stroke();
  }

  const xs = galaxyState.points.map((p) => p.x);
  const ys = galaxyState.points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const mp = galaxyState.morphProgress ?? 1;
  galaxyState.screenPoints = [];

  galaxyState.points.forEach((p) => {
    const targetNx = (p.x - minX) / (maxX - minX + 1e-9);
    const targetNy = (p.y - minY) / (maxY - minY + 1e-9);

    let nx = targetNx;
    let ny = targetNy;
    if (mp < 1 && galaxyState.prevCoords.has(p.doc_id)) {
      const prev = galaxyState.prevCoords.get(p.doc_id);
      nx = prev.nx + (targetNx - prev.nx) * mp;
      ny = prev.ny + (targetNy - prev.ny) * mp;
    }

    const sx = pad + nx * (W - 2 * pad);
    const sy = H - pad - ny * (H - 2 * pad);

    const groupKey = galaxyState.colorMode === "cluster" ? p.cluster_label : p.true_category;
    const isDimmed = galaxyState.activeFilter && galaxyState.activeFilter !== groupKey;
    const baseColor =
      galaxyState.colorMode === "cluster"
        ? PALETTE[p.cluster_id % PALETTE.length]
        : CATEGORY_COLORS[p.true_category] || "#F59E0B";

    ctx.beginPath();
    ctx.arc(sx, sy, isDimmed ? 3 : 4.6, 0, Math.PI * 2);
    ctx.fillStyle = baseColor;
    ctx.globalAlpha = isDimmed ? 0.13 : 0.86;
    ctx.fill();

    if (!isDimmed) {
      galaxyState.screenPoints.push({ sx, sy, point: p });
    }
  });
  ctx.globalAlpha = 1.0;

  // Draw Live Query Star (⭐) and dashed connections to its Top-3 Nearest Neighbors
  if (galaxyState.liveQuery) {
    const projMode = document.getElementById("cluster-proj-select").value;
    const coords = projMode === "pca" ? galaxyState.liveQuery.pca : galaxyState.liveQuery.tsne;
    if (coords) {
      const qx = Math.min(W - pad, Math.max(pad, pad + ((coords.x - minX) / (maxX - minX + 1e-9)) * (W - 2 * pad)));
      const qy = Math.min(H - pad, Math.max(pad, H - pad - ((coords.y - minY) / (maxY - minY + 1e-9)) * (H - 2 * pad)));

      const topNeighbors = (galaxyState.liveQuery.neighbors || []).slice(0, 3);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.85)";
      ctx.lineWidth = 1.7;
      topNeighbors.forEach((nb) => {
        const match = galaxyState.points.find((pt) => pt.doc_id === nb.doc_id);
        if (match) {
          const nx = pad + ((match.x - minX) / (maxX - minX + 1e-9)) * (W - 2 * pad);
          const ny = H - pad - ((match.y - minY) / (maxY - minY + 1e-9)) * (H - 2 * pad);
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(nx, ny);
          ctx.stroke();
        }
      });
      ctx.restore();

      const pulseRadius = 14 + Math.sin(galaxyState.pulsePhase) * 3.5;
      ctx.beginPath();
      ctx.arc(qx, qy, pulseRadius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(16, 185, 129, 0.24)";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#10B981";
      ctx.stroke();

      ctx.font = "16px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("⭐", qx, qy);

      const labelText = `⭐ Query: ${galaxyState.liveQuery.predicted_category} (${galaxyState.liveQuery.confidence_pct.toFixed(1)}%)`;
      ctx.font = "bold 11.5px Plus Jakarta Sans, Inter, sans-serif";
      const textWidth = ctx.measureText(labelText).width;
      const boxW = textWidth + 16;
      const boxH = 22;
      const bx = qx + 16 + boxW > W - 10 ? qx - 16 - boxW : qx + 14;
      const by = Math.max(12, qy - 11);

      ctx.fillStyle = "rgba(9, 9, 11, 0.96)";
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.strokeStyle = "#10B981";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(bx, by, boxW, boxH);

      ctx.fillStyle = "#34D399";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, bx + 8, by + boxH / 2);

      galaxyState.screenPoints.push({
        sx: qx,
        sy: qy,
        point: {
          doc_id: "⭐ LIVE USER QUERY",
          true_category: galaxyState.liveQuery.predicted_category,
          subtopic: `${galaxyState.liveQuery.confidence_pct.toFixed(1)}% Confidence`,
          cluster_label: `Nearest Neighbor: ${topNeighbors[0] ? topNeighbors[0].doc_id : "N/A"}`,
          snippet: galaxyState.liveQuery.text
        }
      });
    }
  }
}

/* =========================================================================
   6. TAB 4: TOPIC DISCOVERY (3x2 GRID + WEIGHTED PILLS) & PUBLIC SANDBOX
   ========================================================================= */
function initTopicsTab() {
  const scopeSelect = document.getElementById("topic-scope-select");
  const countSlider = document.getElementById("topic-count-slider");

  countSlider.addEventListener("input", () => {
    document.getElementById("topic-count-label").textContent = countSlider.value;
  });
  scopeSelect.addEventListener("change", loadTopicsData);
  countSlider.addEventListener("change", loadTopicsData);

  document.getElementById("btn-run-batch").addEventListener("click", runCustomBatch);

  const btnRandBatch = document.getElementById("btn-rand-batch");
  if (btnRandBatch) {
    btnRandBatch.addEventListener("click", () => {
      const domains = Object.keys(RANDOM_DOMAIN_POOL);
      const picked = domains.map((d) => {
        const list = RANDOM_DOMAIN_POOL[d];
        return list[Math.floor(Math.random() * list.length)];
      });
      for (let i = picked.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [picked[i], picked[j]] = [picked[j], picked[i]];
      }
      const batchInput = document.getElementById("batch-input");
      batchInput.value = picked.join("\n");
      document.getElementById("batch-k-input").value = "5";
      flashInput(batchInput);
      runCustomBatch();
    });
  }
}

async function loadTopicsData() {
  const rawScope = document.getElementById("topic-scope-select").value;
  const nTopics = document.getElementById("topic-count-slider").value;

  const isCorpus = rawScope === "corpus";
  const url = isCorpus
    ? `/api/topics?scope=corpus&n_topics=${nTopics}`
    : `/api/topics?scope=category&category=${encodeURIComponent(rawScope)}&n_topics=${nTopics}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    document.getElementById("topic-scope-badge").textContent = data.category;
    animateNumber(document.getElementById("topic-div-score"), data.topic_diversity * 100, 1, "%");
    animateNumber(document.getElementById("topic-coh-score"), data.mean_coherence, 4);

    animateNumber(document.getElementById("kpi-div"), data.topic_diversity * 100, 1, "%");
    const kpiSub = document.getElementById("kpi-div-sub");
    if (kpiSub) {
      kpiSub.textContent = `Cluster c-TF-IDF (${data.scope === "corpus" ? "Corpus" : data.category})`;
    }

    const container = document.getElementById("topics-cards-container");
    container.innerHTML = "";

    data.topics.forEach((t, idx) => {
      const accentColor = PALETTE[idx % PALETTE.length];
      const maxWt = Math.max(...t.keywords.map((k) => k.weight), 0.001);
      const cohScore = Number(t["Semantic Coherence"] || 0.45);
      const cohPct = Math.min(100, Math.max(15, Math.round(cohScore * 130)));

      // Weighted interactive pill chips
      const pillsHtml = t.keywords
        .map((k) => {
          const ratio = k.weight / maxWt;
          const fontSize = (10.5 + ratio * 2.2).toFixed(1);
          const alpha = (0.10 + ratio * 0.20).toFixed(2);
          return `<span class="kw-pill-chip" style="font-size:${fontSize}px; background:rgba(16, 185, 129, ${alpha}); border-color:${accentColor}66;" title="c-TF-IDF weight: ${k.weight.toFixed(4)}">${k.word} <small style="opacity:0.75;">${k.weight.toFixed(2)}</small></span>`;
        })
        .join("");

      // Top-4 keyword bars
      const kwBarsHtml = t.keywords
        .slice(0, 4)
        .map(
          (k) => `
          <div class="kw-bar-row">
            <span class="kw-word">${k.word}</span>
            <div class="kw-track">
              <div class="kw-fill" style="width:${Math.round((k.weight / maxWt) * 100)}%; background:${accentColor}"></div>
            </div>
            <span class="mono" style="font-size:10.5px; color:var(--text-muted);">${k.weight.toFixed(3)}</span>
          </div>
        `
        )
        .join("");

      const card = document.createElement("div");
      card.className = "topic-card";
      card.style.setProperty("--i", idx);
      card.innerHTML = `
        <div class="topic-card-header">
          <div class="topic-card-title">${t["Discovered Theme"]}</div>
          <span class="kpi-badge-mono">${t.Documents} Docs</span>
        </div>
        <div class="coherence-gauge-row">
          <span style="color:var(--text-muted); font-weight:600;">Coherence</span>
          <div class="coherence-track">
            <div class="coherence-fill" style="width:${cohPct}%;"></div>
          </div>
          <code style="color:#34D399; font-weight:700;">${cohScore.toFixed(4)}</code>
        </div>
        <div class="kw-pills-wrap">${pillsHtml}</div>
        <div>${kwBarsHtml}</div>
        <blockquote class="topic-exemplar">
          "${t["Exemplar Document"]}"
        </blockquote>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Topics load error:", err);
  }
}

async function runCustomBatch() {
  const raw = document.getElementById("batch-input").value;
  const k = parseInt(document.getElementById("batch-k-input").value || "3", 10);
  const texts = raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  if (texts.length < 3) return;

  const spinner = document.getElementById("batch-spinner");
  if (spinner) spinner.classList.remove("hidden");

  try {
    const res = await fetch("/api/analyze-custom-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts, n_clusters: k })
    });
    const data = await res.json();
    if (data.error) return;

    const tbody = document.querySelector("#batch-results-table tbody");
    tbody.innerHTML = "";
    data.items.forEach((item, idx) => {
      const tr = document.createElement("tr");
      tr.style.setProperty("--i", idx);
      const domColor = CATEGORY_COLORS[item.predicted_category] || "#F59E0B";
      const confColor = item.is_low_confidence ? "#F59E0B" : domColor;
      tr.innerHTML = `
        <td><code>${item.index}</code></td>
        <td>${item.text}</td>
        <td><span class="domain-pill" data-domain="${item.predicted_category}">${item.predicted_category}</span></td>
        <td>
          <div class="inline-conf-wrap">
            <span class="mono" style="font-size:11.5px; font-weight:700; color:${confColor};">${item.confidence_pct.toFixed(1)}%</span>
            <div class="inline-conf-bar">
              <div class="inline-conf-fill" style="width:${Math.round(item.confidence_pct)}%; background:${confColor};"></div>
            </div>
          </div>
        </td>
        <td><span class="cluster-group-badge">${item.cluster_id}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Batch error:", err);
  } finally {
    if (spinner) spinner.classList.add("hidden");
  }
}

/* =========================================================================
   7. GALLERY LIGHTBOX MODAL
   ========================================================================= */
function initGalleryModal() {
  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img");
  const closeBtn = document.getElementById("modal-close");

  document.querySelectorAll(".gallery-item").forEach((item) => {
    item.addEventListener("click", () => {
      modalImg.src = item.dataset.img;
      modal.classList.remove("hidden");
    });
  });

  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      modal.classList.add("hidden");
    }
  });
}
