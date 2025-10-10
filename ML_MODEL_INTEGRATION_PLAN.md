# ML Model Integration Plan for Browser Extension

## Overview
This document outlines the detailed plan to integrate the existing ML purchase probability model into the current e-commerce browser extension without removing or duplicating existing features.

## Current Extension Features
- Extracts product details (name, price, rating, specs) from e-commerce pages (Amazon and generic)
- Extracts and analyzes Amazon reviews with local sentiment analysis
- Provides product comparison using Gemini API
- UI elements for analysis, best buy recommendations, sentiment analysis, product comparison, and API testing

## Integration Goals
- Add real-time purchase probability predictions and recommendations using the ML model
- Preserve all existing features and UI components
- Ensure seamless user experience with new ML features integrated into the popup and content scripts
- Maintain code modularity and clarity

## Integration Architecture
1. **Local API Server**
   - Run `extension_api.py` as a FastAPI server locally on port 8001
   - Loads the trained ML model (`deployed_model.pkl`)
   - Exposes `/predict_event` endpoint for prediction requests

2. **Extension Content Scripts**
   - Add a new content script `content_predict.js` to:
     - Listen to user events (clicks, scrolls, time spent)
     - Extract features required by the ML model (brand, category_code, price, hour, day_of_week, is_weekend, etc.)
     - Send features as JSON POST requests to the local API `/predict_event`
     - Receive purchase probability and recommendation from API

3. **Popup UI**
   - Update `popup.js` to:
     - Display ML model predictions alongside existing product data
     - Show purchase probability and recommendation badges or messages
     - Allow user to toggle ML features on/off

4. **Manifest Update**
   - Add `content_predict.js` to `manifest.json` content scripts
   - Ensure permissions and host permissions remain intact

## Implementation Steps

### Phase 1: Preparation
- Verify API server runs and serves predictions correctly
- Review existing extension code for event handling and UI updates

### Phase 2: Content Script Enhancement
- Create `content_predict.js` for ML event collection and API communication
- Integrate with existing `content.js` to avoid duplicate event listeners
- Handle API errors gracefully

### Phase 3: Popup UI Update
- Modify `popup.js` to include ML prediction display
- Add UI elements for purchase probability and recommendation
- Ensure existing features remain functional and unchanged

### Phase 4: Manifest and Packaging
- Update `manifest.json` to include new content script
- Test extension loading and functionality

### Phase 5: Testing
- Test API server startup and prediction endpoint
- Test extension event sending and prediction display
- Verify all existing features work as before
- Perform cross-browser testing if applicable

## Documentation and User Guide
- Update README and create this integration plan document
- Provide instructions to start API server and load updated extension
- Include troubleshooting tips

## Future Enhancements
- Session-level feature aggregation
- Model retraining pipeline integration
- Privacy and data consent features

---

This plan ensures the ML model integration enhances the extension without disrupting existing functionality.
