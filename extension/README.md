# Avito Visible Collector (Manifest V3)

This extension reads listing cards that Avito has already rendered in the active browser tab and submits their image and metadata to the configured Visual Search API.

## Install

1. Start the Visual Search API and configure `VISUAL_SEARCH_API_KEY` if the API is exposed outside localhost.
2. Open Chrome and visit `chrome://extensions`.
3. Enable Developer mode.
4. Choose **Load unpacked** and select this `extension` folder.
5. Open the extension popup and enter the API base URL (for example, `http://127.0.0.1:8000` for a local API) and the API key if configured.
6. Save settings. Chrome will ask permission for the API origin.
7. Open an Avito search-results page, wait for cards to load, then click **Отправить видимые объявления**.
8. Scroll normally to load more cards and repeat.

## Scope and limits

- Reads only cards currently visible in the page DOM. It does not navigate pages or query hidden/private endpoints.
- Does not bypass login, CAPTCHA, anti-bot protections, rate limits, or access restrictions.
- The extension needs access to the image host to fetch listing photos. Image hosts can change; if an image request is blocked, the popup reports an error.
- The API key is stored in Chrome extension local storage. It is not a hardware-backed secret and should be treated as exposed to the local browser profile. Use a dedicated, revocable key.
- Use HTTPS for remote API deployments. Plain HTTP is allowed only for localhost.
