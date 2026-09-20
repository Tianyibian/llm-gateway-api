const conversation = document.querySelector("#conversation");
const form = document.querySelector("#composer");
const input = document.querySelector("#query");
const sendButton = document.querySelector("#send-button");
const imageInput = document.querySelector("#image-input");
const attachmentPreview = document.querySelector("#attachment-preview");
const attachmentImage = document.querySelector("#attachment-image");
const attachmentName = document.querySelector("#attachment-name");
const removeAttachmentButton = document.querySelector("#remove-attachment");
const conversationList = document.querySelector("#conversation-list");
const newConversationButton = document.querySelector("#new-conversation");
const refreshConversationsButton = document.querySelector("#refresh-conversations");
const currentConversationTitle = document.querySelector("#current-conversation-title");
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

let selectedImage = null;
let selectedImageUrl = null;
const USER_ID_KEY = "aster_user_id";
const CONVERSATION_ID_KEY = "aster_conversation_id";
const userId = localStorage.getItem(USER_ID_KEY) ?? crypto.randomUUID();
let conversationId = localStorage.getItem(CONVERSATION_ID_KEY);
let conversationRecords = [];
localStorage.setItem(USER_ID_KEY, userId);

const routeLabels = {
  general_search: "General assistance",
  product_search: "Product catalog",
  additional_search: "Additional information needed",
  policy_search: "Policies and support",
  analytics_search: "Snowflake analytics",
  graph_rag_search: "GraphRAG: Neo4j / Microsoft",
  vision_analysis: "Image analysis",
};

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function addUserMessage(text, imageFile = null) {
  const article = document.createElement("article");
  article.className = "message user-message";
  const body = document.createElement("div");
  body.className = "message-body";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  if (imageFile) {
    const upload = document.createElement("span");
    upload.className = "message-meta";
    upload.textContent = `Attached image: ${imageFile.name}`;
    body.append(upload);
  }
  body.append(paragraph);
  article.append(body);
  conversation.append(article);
}

function addAssistantMessage(content = null) {
  const article = document.createElement("article");
  article.className = "message assistant-message";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "A";

  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = "Aster";
  const route = document.createElement("div");
  route.className = "route-badge";
  route.hidden = true;
  const answer = document.createElement("p");
  answer.className = content === null ? "thinking" : "";
  answer.textContent = content ?? "Routing your question";
  const sources = document.createElement("div");
  sources.className = "source-panel";

  const inspector = GraphInspector.create();
  body.append(meta, route, inspector.overview, answer, sources, inspector.panel);
  article.append(avatar, body);
  conversation.append(article);
  return { answer, route, sources, inspector, receivedDelta: false, streamFinished: false };
}

function renderWelcome() {
  conversation.replaceChildren();
  addAssistantMessage(
    "Ask a question and the router will select a branch. Try review themes for Microsoft GraphRAG, then expand Execution details to see scope decisions, retrieval tasks, and evidence. Use full product names for graph questions."
  );

  const suggestions = document.createElement("section");
  suggestions.className = "suggestions";
  suggestions.setAttribute("aria-label", "Suggested questions");
  [
    ["What can you help me with?", "What can you help me with?"],
    ["Show me Philips Hue smart locks and their inventory.", "Find Philips Hue smart locks"],
    ["Can I return an installed smart lock?", "Ask about a return"],
    ["Which five products generated the most revenue in 2025?", "Analyze product revenue"],
    ["Summarize the recurring support themes described in the sampled Eufy Smart Speaker Essential reviews.", "GraphRAG: review themes"],
    ["Compare the setup and connectivity concerns in the sampled Eufy Smart Speaker Essential and Belkin Wemo Security System Pro reviews. Describe where retrieved evidence is insufficient.", "GraphRAG: compare two products"],
  ].forEach(([prompt, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.prompt = prompt;
    button.textContent = label;
    suggestions.append(button);
  });
  conversation.append(suggestions);
}

function startNewConversation() {
  conversationId = null;
  localStorage.removeItem(CONVERSATION_ID_KEY);
  currentConversationTitle.textContent = "New conversation";
  renderConversationList(conversationRecords);
  renderWelcome();
  input.focus();
}

function formatConversationDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function renderConversationList(items) {
  conversationList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-list-status";
    empty.textContent = "No saved conversations yet.";
    conversationList.append(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = `conversation-item${item.id === conversationId ? " active" : ""}`;

    const select = document.createElement("button");
    select.type = "button";
    select.className = "conversation-select";
    select.dataset.conversationAction = "select";
    select.dataset.conversationId = item.id;
    const title = document.createElement("strong");
    title.textContent = item.title || "Untitled conversation";
    const date = document.createElement("span");
    date.textContent = formatConversationDate(item.updated_at);
    select.append(title, date);

    const actions = document.createElement("div");
    actions.className = "conversation-actions";
    const rename = document.createElement("button");
    rename.type = "button";
    rename.dataset.conversationAction = "rename";
    rename.dataset.conversationId = item.id;
    rename.setAttribute("aria-label", `Rename ${item.title || "conversation"}`);
    rename.title = "Rename";
    rename.textContent = "✎";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.dataset.conversationAction = "delete";
    remove.dataset.conversationId = item.id;
    remove.setAttribute("aria-label", `Delete ${item.title || "conversation"}`);
    remove.title = "Delete";
    remove.textContent = "×";
    actions.append(rename, remove);
    row.append(select, actions);
    conversationList.append(row);
  });
}

async function loadConversation(id) {
  const item = conversationRecords.find((record) => record.id === id);
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(id)}/messages?user_id=${encodeURIComponent(userId)}`
  );
  if (!response.ok) throw new Error(`Could not load conversation (${response.status})`);
  const messages = await response.json();

  conversationId = id;
  localStorage.setItem(CONVERSATION_ID_KEY, id);
  currentConversationTitle.textContent = item?.title || "Untitled conversation";
  conversation.replaceChildren();
  messages.forEach((message) => {
    if (message.role === "user") addUserMessage(message.content);
    if (message.role === "assistant") addAssistantMessage(message.content);
  });
  if (!messages.length) renderWelcome();
  renderConversationList(conversationRecords);
  scrollToLatest();
  input.focus();
}

async function loadConversations({ restoreActive = false } = {}) {
  conversationList.innerHTML = '<p class="conversation-list-status">Loading conversations…</p>';
  try {
    const response = await fetch(`/api/users/${encodeURIComponent(userId)}/conversations`);
    if (!response.ok) throw new Error(`Could not load conversations (${response.status})`);
    conversationRecords = await response.json();
    renderConversationList(conversationRecords);

    if (restoreActive && conversationId) {
      if (conversationRecords.some((item) => item.id === conversationId)) {
        await loadConversation(conversationId);
      } else {
        startNewConversation();
      }
    }
  } catch (error) {
    conversationList.innerHTML = '<p class="conversation-list-status">Conversation history is unavailable.</p>';
    console.error(error);
  }
}

async function renameConversation(id) {
  const item = conversationRecords.find((record) => record.id === id);
  const title = window.prompt("Conversation title", item?.title || "");
  if (!title?.trim()) return;
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, title: title.trim() }),
  });
  if (!response.ok) throw new Error(`Could not rename conversation (${response.status})`);
  if (id === conversationId) currentConversationTitle.textContent = title.trim();
  await loadConversations();
}

async function deleteConversation(id) {
  const item = conversationRecords.find((record) => record.id === id);
  if (!window.confirm(`Delete “${item?.title || "this conversation"}” and all of its messages?`)) return;
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(id)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" }
  );
  if (!response.ok) throw new Error(`Could not delete conversation (${response.status})`);
  if (id === conversationId) startNewConversation();
  await loadConversations();
}

function showProducts(container, payload) {
  container.replaceChildren();
  if (!payload.products?.length) return;

  const title = document.createElement("div");
  title.className = "source-title";
  title.textContent = `Matched records · ${payload.sources.join(", ")}`;
  container.append(title);

  payload.products.slice(0, 3).forEach((product) => {
    const card = document.createElement("div");
    card.className = "product-card";
    const name = document.createElement("strong");
    name.textContent = product.product_name;
    const price = document.createElement("strong");
    price.textContent = `$${Number(product.unit_price).toFixed(2)}`;
    const category = document.createElement("span");
    category.textContent = `${product.category} · ${product.supplier}`;
    const stock = document.createElement("span");
    stock.textContent = `${product.units_in_stock} in stock`;
    card.append(name, price, category, stock);
    container.append(card);
  });
}

function showKnowledgeSources(container, payload) {
  container.replaceChildren();
  const documents = payload.documents ?? [];
  if (!documents.length) return;

  const title = document.createElement("div");
  title.className = "source-title";
  title.textContent = "Knowledge Base sources";
  container.append(title);

  const uniqueSources = new Map();
  documents.forEach((source) => {
    const key = `${source.source_path}:${source.chunk_index}`;
    if (!uniqueSources.has(key)) uniqueSources.set(key, source);
  });
  [...uniqueSources.values()].forEach((source, index) => {
    const card = document.createElement("div");
    card.className = "knowledge-source";
    const marker = document.createElement("strong");
    marker.textContent = `[${index + 1}]`;
    const details = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = source.title;
    const meta = document.createElement("span");
    const page = source.page ? ` · page ${source.page}` : "";
    const relevance = Number.isFinite(Number(source.score))
      ? ` · ${Math.round(Number(source.score) * 100)}% match`
      : "";
    meta.textContent = `${source.category ?? source.source_type}${page}${relevance}`;
    details.append(name, meta);
    card.append(marker, details);
    card.title = source.source_file;
    container.append(card);
  });
}

function showAnalyticsRows(container, payload) {
  container.replaceChildren();
  const rows = payload.rows ?? [];
  if (!rows.length) return;

  const title = document.createElement("div");
  title.className = "source-title";
  const duration = Number.isFinite(Number(payload.elapsed_ms))
    ? ` · ${Number(payload.elapsed_ms).toFixed(0)} ms`
    : "";
  title.textContent = `Snowflake result${duration}`;
  container.append(title);

  rows.slice(0, 5).forEach((row) => {
    const card = document.createElement("div");
    card.className = "product-card";
    Object.entries(row).forEach(([key, value]) => {
      const field = document.createElement("span");
      field.textContent = `${key.replaceAll("_", " ")}: ${value}`;
      card.append(field);
    });
    container.append(card);
  });
}

function handleEvent(eventName, payload, message) {
  if (eventName === "metadata" && payload.conversation_id) {
    conversationId = payload.conversation_id;
    localStorage.setItem(CONVERSATION_ID_KEY, conversationId);
  } else if (eventName === "route") {
    GraphInspector.record(message.inspector, "Router", payload);
    message.route.hidden = false;
    const confidence = Math.round(payload.confidence * 100);
    const provider = payload.provider ? ` · ${payload.provider}` : "";
    message.route.textContent = `${routeLabels[payload.route] ?? payload.route}${provider} · ${confidence}%`;
    message.route.title = payload.reason;
    message.answer.textContent = payload.route === "vision_analysis" ? "Reading the image"
      : payload.route === "graph_rag_search" ? "Checking GraphRAG scope and retrieving evidence. This may take a few minutes…"
      : "Preparing a response…";
  } else if (eventName === "guardrail") {
    GraphInspector.record(message.inspector, "Guardrail", payload);
  } else if (eventName === "clarification") {
    GraphInspector.record(message.inspector, "Clarification guardrail", payload);
  } else if (eventName === "supervisor") {
    GraphInspector.record(message.inspector, "Supervisor", payload);
  } else if (eventName === "agent") {
    GraphInspector.record(message.inspector, "Subagent", payload);
    if (!message.receivedDelta) message.answer.textContent = payload.stage === "started"
      ? `${payload.agent} is working on its assigned question…`
      : `${payload.agent}: ${payload.status}. Preparing the next step…`;
  } else if (eventName === "answer_generation") {
    GraphInspector.record(message.inspector, "Answer generation", payload);
    if (!message.receivedDelta) {
      message.answer.textContent = payload.stage === "map"
        ? `Reading evidence (${payload.completed}/${payload.total})…`
        : payload.stage === "reduce" ? "Combining evidence into an answer…"
        : payload.stage === "validate_citations" ? "Checking answer citations…"
        : payload.stage === "complete" ? "Answer validated. Preparing delivery…"
        : "Could not produce a validated answer.";
    }
  } else if (eventName === "sources") {
    if (payload.backend === "snowflake") showAnalyticsRows(message.sources, payload);
    else if (payload.documents?.length) showKnowledgeSources(message.sources, payload);
    else showProducts(message.sources, payload);
  } else if (eventName === "delta") {
    message.answer.classList.remove("thinking");
    if (!message.receivedDelta) {
      message.answer.textContent = "";
      message.receivedDelta = true;
    }
    message.answer.textContent += payload.content;
  } else if (eventName === "error") {
    message.streamFinished = true;
    message.answer.classList.remove("thinking");
    message.answer.classList.add("error-text");
    message.answer.textContent = payload.message ?? "The assistant request failed.";
  } else if (eventName === "done") {
    message.streamFinished = true;
  }
  scrollToLatest();
}

async function readEventStream(response, message) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      let eventName = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) eventName = line.slice(7);
        if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      handleEvent(eventName, payload, message);
    }
    if (done) break;
  }
  if (!message.streamFinished) throw new Error("The response stream ended before completion.");
}

async function refreshGraphStatus() {
  const badge = document.querySelector("#graph-index-status");
  const button = document.querySelector("#refresh-graph-status");
  button.disabled = true;
  badge.textContent = "Checking graph index…";
  try {
    const response = await fetch("/api/graphrag/status");
    if (!response.ok) throw new Error("Status request failed");
    const status = await response.json();
    badge.textContent = status.ready
      ? `Graph index ready · ${status.counts.entities} entities`
      : "Graph index unavailable";
    badge.title = status.ready
      ? `Available: ${status.available_modes.join(", ")}. Local index readiness, not a model connectivity check.`
      : "Check Microsoft GraphRAG configuration and index verification. Other routes may still work.";
  } catch {
    badge.textContent = "Graph status unavailable";
    badge.title = "Start the API server, then refresh.";
  } finally {
    button.disabled = false;
  }
}

function clearAttachment() {
  selectedImage = null;
  imageInput.value = "";
  attachmentPreview.hidden = true;
  attachmentImage.removeAttribute("src");
  attachmentName.textContent = "";
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = null;
}

function selectAttachment(file) {
  if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
    window.alert("Choose a PNG, JPEG, WEBP, or non-animated GIF image.");
    clearAttachment();
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    window.alert("The image must be 10 MB or smaller.");
    clearAttachment();
    return;
  }

  clearAttachment();
  selectedImage = file;
  selectedImageUrl = URL.createObjectURL(file);
  attachmentImage.src = selectedImageUrl;
  attachmentName.textContent = file.name;
  attachmentPreview.hidden = false;
  input.placeholder = "What would you like to know about this image?";
  input.focus();
}

async function sendQuery(query) {
  const imageFile = selectedImage;
  const text = query.trim() || (imageFile ? "What is in this image?" : "");
  if (!text || sendButton.disabled) return;
  const isNewConversation = !conversationId;
  if (isNewConversation) currentConversationTitle.textContent = text.slice(0, 80);

  document.querySelector(".suggestions")?.remove();
  addUserMessage(text, imageFile);
  const message = addAssistantMessage();
  input.value = "";
  input.style.height = "auto";
  clearAttachment();
  input.placeholder = "Ask about products, inventory, or anything else…";
  sendButton.disabled = true;
  newConversationButton.disabled = true;
  refreshConversationsButton.disabled = true;
  scrollToLatest();

  try {
    let response;
    if (imageFile) {
      const formData = new FormData();
      formData.append("query", text);
      formData.append("user_id", userId);
      if (conversationId) formData.append("conversation_id", conversationId);
      formData.append("image", imageFile);
      response = await fetch("/api/assistant", {
        method: "POST",
        body: formData,
      });
    } else {
      response = await fetch("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          user_id: userId,
          conversation_id: conversationId,
        }),
      });
    }
    if (!response.ok || !response.body) {
      const detail = await response.text();
      throw new Error(detail || `Request failed with ${response.status}`);
    }
    await readEventStream(response, message);
    await loadConversations();
  } catch (error) {
    message.answer.classList.remove("thinking");
    message.answer.classList.add("error-text");
    message.answer.textContent = "I could not reach the assistant. Confirm that the API and model are running.";
    console.error(error);
  } finally {
    sendButton.disabled = false;
    newConversationButton.disabled = false;
    refreshConversationsButton.disabled = false;
    input.focus();
    scrollToLatest();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuery(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

conversation.addEventListener("click", (event) => {
  const button = event.target.closest("[data-prompt]");
  if (button) sendQuery(button.dataset.prompt ?? "");
});

conversationList.addEventListener("click", async (event) => {
  if (sendButton.disabled) return;
  const button = event.target.closest("[data-conversation-action]");
  if (!button) return;
  const { conversationAction: action, conversationId: id } = button.dataset;
  if (!id) return;
  try {
    if (action === "select") await loadConversation(id);
    if (action === "rename") await renameConversation(id);
    if (action === "delete") await deleteConversation(id);
  } catch (error) {
    window.alert("The conversation action failed. Check that the API and database are running.");
    console.error(error);
  }
});

newConversationButton.addEventListener("click", startNewConversation);
document.querySelector("#refresh-graph-status").addEventListener("click", refreshGraphStatus);
refreshConversationsButton.addEventListener("click", () => loadConversations());

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files ?? [];
  if (file) selectAttachment(file);
});

removeAttachmentButton.addEventListener("click", () => {
  clearAttachment();
  input.placeholder = "Ask about products, inventory, or anything else…";
});

loadConversations({ restoreActive: true });
refreshGraphStatus();
