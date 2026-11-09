# Security notes

- Never commit API keys, tunnel URLs intended to be private, cookies, browser profiles, or personal account data.
- Set `VISUAL_SEARCH_API_KEY` for any endpoint exposed to the internet.
- Set `CORS_ALLOWED_ORIGINS` to explicit trusted origins. An empty value disables browser CORS rather than allowing every origin.
- Uploaded images are limited by `MAX_IMAGE_BYTES` (default 12 MiB) and `MAX_IMAGE_PIXELS` (default 25 million pixels).
- The Avito collector must only read cards rendered in the user's ordinary browser session. It must not bypass CAPTCHA, authentication, anti-bot controls, or rate limits.
- Treat listing titles, descriptions, image URLs, and page content as untrusted data.
- The browser extension stores its API key in Chrome extension local storage. Treat it as exposed to the local browser profile and use a dedicated, revocable key.
- Rotate any API key that has ever been pasted into a chat, notebook output, source file, or extension bundle. Do not reuse it.
- Restrict public deployment behind HTTPS and a reverse proxy/rate limiter. The shared API key is basic access control, not a full user-management system.
