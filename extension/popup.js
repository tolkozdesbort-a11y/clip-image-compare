const apiInput = document.getElementById("api");
const keyInput = document.getElementById("key");
const statusEl = document.getElementById("status");
const saveButton = document.getElementById("save");
const collectButton = document.getElementById("collect");

function show(message) { statusEl.textContent = message; }

async function loadSettings() {
  const saved = await chrome.storage.local.get(["apiUrl", "apiKey"]);
  apiInput.value = saved.apiUrl || "";
  keyInput.value = saved.apiKey || "";
}
loadSettings();

saveButton.addEventListener("click", async () => {
  const apiUrl = apiInput.value.trim().replace(/\/+$/, "");
  const apiKey = keyInput.value.trim();
  let parsed;
  try { parsed = new URL(apiUrl); }
  catch { show("Укажи корректный адрес API, включая https://"); return; }
  if (!["https:", "http:"].includes(parsed.protocol)) {
    show("Поддерживаются только HTTP и HTTPS."); return;
  }
  if (parsed.protocol === "http:" && !["localhost", "127.0.0.1"].includes(parsed.hostname)) {
    show("Для удалённого сервера используй HTTPS."); return;
  }
  const originPattern = parsed.origin + "/*";
  const granted = await chrome.permissions.request({ origins: [originPattern] });
  if (!granted) { show("Chrome не выдал разрешение на адрес API."); return; }
  await chrome.storage.local.set({ apiUrl, apiKey });
  show("Настройки сохранены. Теперь открой Avito и нажми «Отправить видимые объявления».");
});

collectButton.addEventListener("click", async () => {
  collectButton.disabled = true;
  show("Сканирую карточки на текущей странице…");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !/^https:\/\/(www\.)?avito\.ru\//.test(tab.url || "")) {
      show("Сначала открой страницу Avito."); return;
    }
    const saved = await chrome.storage.local.get(["apiUrl", "apiKey"]);
    if (!saved.apiUrl) { show("Сначала сохрани адрес API."); return; }
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_VISIBLE_LISTINGS" });
    if (!response?.items?.length) { show("Карточки не найдены. Прокрути страницу, чтобы объявления загрузились."); return; }
    show("Найдено карточек: " + response.items.length + ". Отправляю изображения…");
    const result = await chrome.runtime.sendMessage({ type: "INGEST_LISTINGS", items: response.items });
    show(
      "Готово.\nНайдено: " + response.items.length +
      "\nДобавлено: " + (result?.sent || 0) +
      "\nДубликатов: " + (result?.duplicates || 0) +
      "\nОшибок: " + (result?.errors || 0) +
      (result?.details?.length ? "\n\nПодробности:\n" + result.details.slice(0, 5).join("\n") : "")
    );
  } catch (error) {
    show("Ошибка: " + (error?.message || String(error)));
  } finally {
    collectButton.disabled = false;
  }
});
