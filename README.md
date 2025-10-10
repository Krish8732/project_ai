# E-commerce Analyzer with ML Purchase Prediction

A Chrome extension that analyzes e-commerce pages, provides product comparisons, sentiment analysis, and now includes real-time ML-powered purchase probability predictions.

## Features

### Existing Features
- **Product Analysis**: Extracts product details (name, price, rating, specs) from e-commerce pages
- **Amazon Review Sentiment**: Analyzes Amazon product reviews with local sentiment analysis
- **Product Comparison**: Compare products using Gemini AI or enhanced fallback data
- **Best Buy Recommendations**: Find the best product within your budget range

### New ML Features
- **Real-time Purchase Prediction**: Uses machine learning to predict purchase probability based on user behavior
- **Personalized Recommendations**: Get high/medium/low purchase recommendations
- **Behavioral Analysis**: Analyzes clicks, scroll depth, time spent, and other engagement metrics

## Installation

1. **Clone or download** this repository
2. **Start the ML API Server**:
   ```bash
   python extension_api.py
   ```
   The server will run on `http://127.0.0.1:8001`

3. **Load the Extension in Chrome**:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select the `webpatternrecognition` folder

4. **Test the Extension**:
   - Navigate to any e-commerce site (Amazon, etc.)
   - Click the extension icon
   - Use the various analysis features
   - The ML prediction will appear automatically as you interact with the page

## How It Works

### ML Model Integration
- The extension collects user interaction data (clicks, scrolls, time spent)
- Extracts product features (price, brand, category, time features)
- Sends data to the local FastAPI server
- Receives real-time purchase probability predictions
- Displays recommendations in the popup UI

### Data Privacy
- All processing happens locally on your machine
- No user data is sent to external servers
- ML model runs offline

## Usage

1. **Basic Analysis**: Click "Analyze E-commerce Page" to extract products
2. **Budget Filtering**: Set min/max price and find best buy
3. **Sentiment Analysis**: Works on Amazon product pages
4. **Product Comparison**: Compare two products side-by-side
5. **ML Predictions**: View real-time purchase probability as you browse

## Troubleshooting

### ML Predictions Not Showing
- Ensure the API server is running (`python extension_api.py`)
- Check browser console for errors
- Verify the extension is loaded and active

### Extension Not Loading
- Check `manifest.json` for syntax errors
- Ensure all required files are present
- Try reloading the extension in Chrome

### API Server Issues
- Make sure port 8001 is not blocked
- Check if `deployed_model.pkl` exists
- Run `python test_api_server.py` to test connectivity

## Development

### Project Structure
```
├── extension_api.py          # FastAPI server for ML predictions
├── deployed_model.pkl        # Trained ML model
├── test_api_server.py       # API testing script
├── webpatternrecognition/
│   ├── manifest.json         # Extension manifest
│   ├── popup.html           # Extension popup UI
│   ├── popup.js             # Main popup logic
│   ├── content.js           # Product extraction
│   ├── content_predict.js   # ML event collection
│   └── icon.png             # Extension icon
├── ML_MODEL_INTEGRATION_PLAN.md  # Integration documentation
└── README.md                # This file
```

### Adding New Features
- Modify `content_predict.js` for new event types
- Update `popup.js` for new UI elements
- Add new endpoints to `extension_api.py` if needed

## Requirements
- Python 3.7+
- Chrome browser
- Required Python packages: fastapi, uvicorn, joblib, pandas, scikit-learn

## License
This project is for educational purposes. Use responsibly and in accordance with website terms of service.

## Contributing
Feel free to submit issues and enhancement requests!
