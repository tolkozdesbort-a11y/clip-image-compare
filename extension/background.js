chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "INGEST_LISTINGS") return;
  (async () => {
    const settings = await chrome.storage.local.get(["apiUrl", "apiKey"]);
    if (!settings.apiUrl) throw new Error("API address is not configured.");
    const headers = settings.apiKey ? { "X-API-Key": settings.apiKey } : {};
    let sent = 0, duplicates = 0, errors = 0;
    const details = [];
    for (const item of (message.items || [])) {
      try {
        const imageResponse = await fetch(item.image_url, { credentials: "omit" });
        if (!imageResponse.ok) throw new Error("Image HTTP " + imageResponse.status);
        const blob = await imageResponse.blob();
        if (!blob.type.startsWith("image/")) throw new Error("URL did not return an image.");
        const form = new FormData();
        form.append("image", blob, "listing-image");
        form.append("metadata", JSON.stringify(item));
        const response = await fetch(settings.apiUrl + "/ingest", {
          method: "POST",
          headers,
          body: form
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          const detail = typeof payload.detail === "string" ? payload.detail : "API HTTP " + response.status;
          throw new Error(detail);
        }
        if (payload.status === "duplicate") duplicates++;
        else sent++;
      } catch (error) {
        errors++;
        if (details.length < 10) details.push((item.title || item.listing_url || "Listing") + ": " + (error?.message || String(error)));
      }
    }
    return { sent, duplicates, errors, details };
  })().then(sendResponse).catch(error => sendResponse({ sent: 0, duplicates: 0, errors: (message.items || []).length, details: [error.message] }));
  return true;
});
