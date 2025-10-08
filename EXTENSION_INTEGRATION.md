# Browser Extension Integration (real-time predictions)

This document shows how to connect your browser extension to the ML model served by this repo. The repo now contains `extension_api.py` which exposes a CORS-enabled endpoint `/predict_event` that accepts a JSON map of features and returns a purchase probability and recommendation.

Steps summary:
1. Start the extension API server locally (serves on port 8001 by default).
2. Update your extension's `content.js` (or background script) to POST user events to `/predict_event`.
3. Use the response to update the extension UI or trigger actions.

1) Start the API server (run in your project root)
```powershell
.\.venv\Scripts\Activate.ps1
python .\extension_api.py
```

2) Example `fetch` snippet to paste into your extension (`content.js` or `background.js`):

```javascript
// Example: collect an event and POST it to the extension API
function sendEventToModel(eventFeatures) {
  // eventFeatures is a plain object mapping feature name -> value
  fetch('http://127.0.0.1:8001/predict_event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(eventFeatures),
  })
  .then(resp => resp.json())
  .then(data => {
    console.log('Model response', data);
    // Update popup or badge or take action based on data.purchase_probability
    // e.g., send message to popup:
    // chrome.runtime.sendMessage({ type: 'MODEL_PREDICTION', payload: data });
  })
  .catch(err => console.error('Model API error', err));
}

// Example usage: call when user clicks a product or scrolls
document.addEventListener('click', e => {
  // collect features from the page (example keys: 'brand', 'category_code', 'price', 'hour', 'day_of_week', 'is_weekend')
  const features = {
    brand: 'example_brand',
    category_code: 'example.category',
    price: 49.99,
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    is_weekend: ([0,6].includes(new Date().getDay()) ? 1 : 0)
  };
  sendEventToModel(features);
});
```

3) Notes and tips
- The extension and the API run on your machine. For production, host `extension_api.py` on a server reachable from extension clients (use HTTPS and API keys).
- Keep the feature names and types aligned with `deployment_package['feature_names']` saved in `deployed_model.pkl`.
- If you need to enrich events with additional features (session-level aggregates), the extension can collect and send them as well.

If you want, I can patch your extension files directly to add this `fetch` call (tell me which file to modify: `content.js` or `background.js`) and implement message passing between the content script and popup.

4) To inject a floating button automatically, add the new content script `content_predict.js` into your extension's manifest. Example manifest fragment (v2/v3 compatible):

```json
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content_predict.js"],
      "run_at": "document_idle"
    }
  ]
```

Place `content_predict.js` in your extension folder (I added it to the attached extension at `webpatternrecognition/content_predict.js`). Then reload the unpacked extension in the browser.

