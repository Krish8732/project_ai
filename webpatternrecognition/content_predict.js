// ML Model Integration Content Script v2.0
// Using chrome.storage for communication

console.log('🚀 ML Content Script v2.0: Starting on', window.location.href);

// Simple test prediction
async function testPrediction() {
  try {
    console.log('🔍 ML: Sending test prediction request...');

    const testData = {
      price: 99.99,
      hour: 14,
      day_of_week: 2,
      is_weekend: 0,
      brand: 'test',
      category_code: 'electronics.test',
      total_events_x: 1
    };

    const response = await fetch('http://127.0.0.1:8001/predict_event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testData)
    });

    console.log('🔍 ML: Response status:', response.status);

    if (response.ok) {
      const data = await response.json();
      console.log('🔍 ML: Prediction received:', data);

      // Store in chrome storage for popup to read
      chrome.storage.local.set({
        'ml_prediction': data,
        'ml_prediction_timestamp': Date.now()
      }, () => {
        console.log('🔍 ML: Prediction stored in chrome.storage');
      });

    } else {
      console.error('🔍 ML: API request failed:', response.status);
    }
  } catch (error) {
    console.error('🔍 ML: Error:', error);
  }
}

// Trigger immediately for testing
setTimeout(() => {
  console.log('🔍 ML: Triggering test prediction...');
  testPrediction();
}, 1000);

// Also trigger on any click for testing
document.addEventListener('click', () => {
  console.log('🔍 ML: Click detected, sending prediction...');
  testPrediction();
});

console.log('🔍 ML Content Script: Loaded successfully');
