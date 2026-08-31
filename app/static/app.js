const conversation = document.querySelector("#conversation");
const form = document.querySelector("#composer");
const input = document.querySelector("#query");
const sendButton = document.querySelector("#send-button");
const imageInput = document.querySelector("#image-input");
const attachmentPreview = document.querySelector("#attachment-preview");
const attachmentImage = document.querySelector("#attachment-image");
const attachmentName = document.querySelector("#attachment-name");
const removeAttachmentButton = document.querySelector("#remove-attachment");
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

let selectedImage = null;
let selectedImageUrl = null;
const USER_ID_KEY = "aster_user_id";
const CONVERSATION_ID_KEY = "aster_conversation_id";
const userId = localStorage.getItem(USER_ID_KEY) ?? crypto.randomUUID();
let conversationId = localStorage.getItem(CONVERSATION_ID_KEY);
localStorage.setItem(USER_ID_KEY, userId);

const routeLabels = {
  general_search: "General assistance",
  product_search: "Product catalog",
  return_search: "Return policy",
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

function addAssistantMessage() {
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
  answer.className = "thinking";
  answer.textContent = "Routing your question";
  const sources = document.createElement("div");
  sources.className = "source-panel";

  body.append(meta, route, answer, sources);
  article.append(avatar, body);
  conversation.append(article);
  return { answer, route, sources };
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

function handleEvent(eventName, payload, message) {
  if (eventName === "metadata" && payload.conversation_id) {
    conversationId = payload.conversation_id;
    localStorage.setItem(CONVERSATION_ID_KEY, conversationId);
  } else if (eventName === "route") {
    message.route.hidden = false;
    const confidence = Math.round(payload.confidence * 100);
    const provider = payload.provider ? ` · ${payload.provider}` : "";
    message.route.textContent = `${routeLabels[payload.route] ?? payload.route}${provider} · ${confidence}%`;
    message.route.title = payload.reason;
    message.answer.textContent = payload.route === "vision_analysis" ? "Reading the image" : "";
    message.answer.classList.remove("thinking");
  } else if (eventName === "sources") {
    showProducts(message.sources, payload);
  } else if (eventName === "delta") {
    message.answer.classList.remove("thinking");
    if (["Routing your question", "Reading the image"].includes(message.answer.textContent)) {
      message.answer.textContent = "";
    }
    message.answer.textContent += payload.content;
  } else if (eventName === "error") {
    message.answer.classList.remove("thinking");
    message.answer.classList.add("error-text");
    message.answer.textContent = payload.message ?? "The assistant request failed.";
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

  document.querySelector(".suggestions")?.remove();
  addUserMessage(text, imageFile);
  const message = addAssistantMessage();
  input.value = "";
  input.style.height = "auto";
  clearAttachment();
  input.placeholder = "Ask about products, inventory, or anything else…";
  sendButton.disabled = true;
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
  } catch (error) {
    message.answer.classList.remove("thinking");
    message.answer.classList.add("error-text");
    message.answer.textContent = "I could not reach the assistant. Confirm that the API and model are running.";
    console.error(error);
  } finally {
    sendButton.disabled = false;
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

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendQuery(button.dataset.prompt ?? ""));
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files ?? [];
  if (file) selectAttachment(file);
});

removeAttachmentButton.addEventListener("click", () => {
  clearAttachment();
  input.placeholder = "Ask about products, inventory, or anything else…";
});
