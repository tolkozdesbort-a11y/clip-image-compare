function visible(element) {
  const rect = element.getBoundingClientRect();
  const style = getComputedStyle(element);
  return rect.width > 0 && rect.height > 0 &&
    rect.bottom > 0 && rect.right > 0 &&
    rect.top < window.innerHeight && rect.left < window.innerWidth &&
    style.visibility !== "hidden" && style.display !== "none";
}

function collectVisibleListings() {
  const cards = [...document.querySelectorAll('[data-marker="item"]')].filter(visible);
  const found = new Map();
  for (const card of cards) {
    const link = card.querySelector('a[href*="_"] , a[href*="/"]');
    const image = [...card.querySelectorAll("img")].find(img => img.currentSrc || img.src);
    if (!link || !image) continue;
    const listingUrl = new URL(link.getAttribute("href"), location.origin).href;
    const imageUrl = image.currentSrc || image.src;
    if (!/^https?:/.test(imageUrl)) continue;
    const title = card.querySelector('[data-marker="item-title"]')?.textContent?.trim()
      || image.alt?.trim() || "";
    const price = card.querySelector('[data-marker="item-price"]')?.textContent?.trim() || "";
    const match = listingUrl.match(/_(\d{6,})(?:\?|$)/);
    const listingId = match?.[1] || "";
    found.set(listingUrl + "|" + imageUrl, {
      listing_id: listingId,
      title,
      price,
      listing_url: listingUrl,
      image_url: imageUrl
    });
  }
  return [...found.values()];
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "GET_VISIBLE_LISTINGS") {
    sendResponse({ items: collectVisibleListings() });
  }
});
