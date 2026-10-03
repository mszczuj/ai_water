"use strict";

const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const conversation = document.querySelector("#conversation");
const status = document.querySelector("#status");
const devStopButton = document.querySelector("#dev-stop");

let history = [];
let model = "qwen3:4b";
let think = "on";
let controller = null;
let webdev = false;

function scrollToPrompt() {
  requestAnimationFrame(() => form.scrollIntoView({ behavior: "smooth", block: "center" }));
}

function element(tag, className, text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function addTurn(userText) {
  const turn = element("article", "turn");
  const userLine = element("div", "user-line");
  userLine.append(
    element("span", "prompt-mark", ">>"),
    element("p", "user-text", userText),
  );

  const thinkingSection = element("section", "thinking");
  const thinkingTitle = element("div", "thinking-title", "myślenie");
  const thinkingContent = element("pre", "thinking-content streaming");
  thinkingSection.append(thinkingTitle, thinkingContent);

  const answerContent = element("pre", "answer-content");
  turn.append(userLine, thinkingSection, answerContent);
  conversation.append(turn);
  scrollToPrompt();
  return { turn, thinkingSection, thinkingContent, answerContent };
}

function showError(view, message) {
  view.turn.append(element("p", "error-message", message));
  scrollToPrompt();
}

function consumeLine(line, view, result) {
  if (!line.trim()) return;
  let chunk;
  try {
    chunk = JSON.parse(line);
  } catch {
    throw new Error("Backend zwrócił nieprawidłowy fragment JSON.");
  }
  if (chunk.error) throw new Error(chunk.error);
  if (chunk.route) {
    result.remember = chunk.route.remember !== false;
    result.matched = chunk.route.matched === true;
  }

  const thinkingChunk = chunk.message?.thinking || "";
  const answerChunk = chunk.message?.content || "";
  if (thinkingChunk) {
    result.thinking += thinkingChunk;
    view.thinkingContent.textContent += thinkingChunk;
  }
  if (answerChunk) {
    result.answer += answerChunk;
    view.answerContent.textContent += answerChunk;
    view.thinkingContent.classList.remove("streaming");
    view.answerContent.classList.add("streaming");
  }
  scrollToPrompt();
}

async function streamChat(messages, view) {
  controller = new AbortController();
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, think, messages }),
    signal: controller.signal,
  });

  if (!response.ok) {
    let message = `Błąd HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data.error) message = data.error;
    } catch {
      // Pozostaw ogólny komunikat HTTP.
    }
    throw new Error(message);
  }
  if (!response.body) throw new Error("Przeglądarka nie udostępniła strumienia odpowiedzi.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const result = { thinking: "", answer: "", remember: true, matched: false };
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) consumeLine(line, view, result);
    if (done) break;
  }
  if (buffer.trim()) consumeLine(buffer, view, result);
  return result;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || controller) return;

  const view = addTurn(text);
  const requestMessages = [...history, { role: "user", content: text }];
  promptInput.value = "";
  promptInput.style.height = "auto";
  promptInput.disabled = true;
  devStopButton.hidden = !webdev;
  status.textContent = "Generowanie odpowiedzi";

  try {
    const result = await streamChat(requestMessages, view);
    view.thinkingContent.classList.remove("streaming");
    view.answerContent.classList.remove("streaming");
    if (!result.thinking) view.thinkingSection.remove();
    if (!result.answer) throw new Error("Model nie zwrócił odpowiedzi.");
    if (result.remember) {
      history = [...requestMessages, { role: "assistant", content: result.answer }];
    }
    status.textContent = "Gotowy";
  } catch (error) {
    view.thinkingContent.classList.remove("streaming");
    view.answerContent.classList.remove("streaming");
    showError(
      view,
      error.name === "AbortError" ? "generowanie zatrzymane" : error.message || String(error),
    );
    status.textContent = "Błąd";
  } finally {
    controller = null;
    devStopButton.hidden = true;
    promptInput.disabled = false;
    promptInput.focus();
    scrollToPrompt();
  }
});

devStopButton.addEventListener("click", () => {
  if (webdev && controller) controller.abort();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  } else if (event.key === "Escape" && controller) {
    controller.abort();
  }
});

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 224)}px`;
});

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const config = await response.json();
    if (config.model) model = config.model;
    if (config.think) think = config.think;
    webdev = config.webdev === true;
  } catch {
    status.textContent = "Backend niedostępny";
  }
}

loadConfig();
