const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const latencyValue = document.getElementById("latencyValue");
const healthBadge = document.getElementById("healthBadge");
const activityLog = document.getElementById("activityLog");

const EDGE_MAP = {
  user: ["user-customer"],
  customer: ["user-customer", "customer-law", "customer-registry"],
  registry: ["customer-registry", "law-registry"],
  law: ["customer-law", "law-registry", "law-tax", "law-compliance"],
  tax: ["law-tax"],
  compliance: ["law-compliance"],
};

const activeCounts = {};
let eventSource = null;

function setHealth(ok) {
  healthBadge.textContent = ok ? "Agents online" : "Agents offline";
  healthBadge.className = `badge ${ok ? "badge-ok" : "badge-error"}`;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    setHealth(data.customer_agent);
  } catch {
    setHealth(false);
  }
}

function appendMessage(role, text, meta = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  if (meta) {
    const m = document.createElement("div");
    m.className = "msg-meta";
    m.textContent = meta;
    wrap.appendChild(m);
  }
  const body = document.createElement("div");
  body.textContent = text;
  wrap.appendChild(body);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function logActivity(text, isActive = false) {
  const line = document.createElement("div");
  line.className = `line${isActive ? " active" : ""}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  activityLog.prepend(line);
  while (activityLog.children.length > 30) {
    activityLog.removeChild(activityLog.lastChild);
  }
}

function setNodeState(agent, state) {
  const node = document.querySelector(`.node[data-agent="${agent}"]`);
  if (!node) return;
  node.classList.remove("active", "done");
  if (state) node.classList.add(state);
}

function setEdgeActive(agent, on) {
  const keys = EDGE_MAP[agent] || [];
  keys.forEach((key) => {
    const edge = document.querySelector(`.edge[data-edge="${key}"]`);
    if (edge) edge.classList.toggle("active", on);
  });
}

function bumpActive(agent, delta) {
  activeCounts[agent] = (activeCounts[agent] || 0) + delta;
  const count = activeCounts[agent];
  if (count > 0) {
    setNodeState(agent, "active");
    setEdgeActive(agent, true);
  } else {
    setNodeState(agent, "done");
    setEdgeActive(agent, false);
    setTimeout(() => {
      if ((activeCounts[agent] || 0) <= 0) setNodeState(agent, null);
    }, 1200);
  }
}

function resetGraph() {
  Object.keys(activeCounts).forEach((k) => delete activeCounts[k]);
  document.querySelectorAll(".node").forEach((n) => n.classList.remove("active", "done"));
  document.querySelectorAll(".edge").forEach((e) => e.classList.remove("active"));
  activityLog.innerHTML = "";
}

function handleTraceEvent(event) {
  const { agent, status, detail } = event;

  if (agent === "system" && status === "completed") {
    latencyValue.textContent = `${event.latency_s}s`;
    appendMessage("bot", event.answer, `Latency: ${event.latency_s}s`);
    sendBtn.disabled = false;
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    logActivity("Request completed");
    return;
  }

  if (agent === "system" && status === "error") {
    appendMessage("bot", `Lỗi: ${detail}`);
    sendBtn.disabled = false;
    if (eventSource) eventSource.close();
    logActivity(`ERROR: ${detail}`);
    return;
  }

  const label = agent.charAt(0).toUpperCase() + agent.slice(1);
  if (status === "started") {
    bumpActive(agent, 1);
    logActivity(`${label} started — ${detail || ""}`, true);
  } else if (status === "completed") {
    bumpActive(agent, -1);
    logActivity(`${label} completed — ${detail || ""}`);
  }
}

function subscribeTrace(traceId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/trace/${traceId}/stream`);
  eventSource.onmessage = (msg) => {
    try {
      handleTraceEvent(JSON.parse(msg.data));
    } catch (e) {
      console.error(e);
    }
  };
  eventSource.onerror = () => {
    logActivity("SSE connection error");
  };
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  resetGraph();
  latencyValue.textContent = "…";
  sendBtn.disabled = true;
  appendMessage("user", question);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const { trace_id } = await res.json();
    subscribeTrace(trace_id);
    logActivity(`Trace ${trace_id.slice(0, 8)}…`);
  } catch (err) {
    appendMessage("bot", `Không gửi được câu hỏi: ${err.message}`);
    sendBtn.disabled = false;
  }
});

checkHealth();
setInterval(checkHealth, 15000);
