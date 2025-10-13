// ML Model Integration Content Script v4.0
// Dynamic user behavior tracking for purchase prediction

console.log('🚀 ML Content Script v4.0: Starting on', window.location.href);

// Session tracking variables
let sessionData = {
  startTime: Date.now(),
  totalEvents: 0,
  scrollEvents: 0,
  clickEvents: 0,
  timeOnPage: 0,
  reviewsViewed: 0,
  imagesViewed: 0,
  addToCartClicked: false,
  buyNowClicked: false,
  sharedProduct: false,
  likedProduct: false,
  lastPredictionTime: 0
};

// Load existing session data from storage
chrome.storage.local.get(['session_data'], (result) => {
  if (result.session_data) {
    sessionData = { ...sessionData, ...result.session_data };
    console.log('📊 Loaded existing session data:', sessionData);
  }
});

// Extract product data from Amazon page
function extractAmazonProductData() {
  const data = {
    price: null,
    brand: null,
    category_code: null,
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    is_weekend: [0, 6].includes(new Date().getDay()) ? 1 : 0
  };

  try {
    // Extract price - try multiple selectors for Amazon
    const priceSelectors = [
      '[data-cy="price-recipe"] .a-price .a-offscreen',
      '.a-price .a-offscreen',
      '#priceblock_ourprice',
      '#priceblock_dealprice',
      '.a-color-price',
      '[data-cy="price-recipe"]'
    ];

    for (const selector of priceSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        const priceText = element.textContent || element.innerText;
        const priceMatch = priceText.match(/\$?(\d+(?:\.\d{2})?)/);
        if (priceMatch) {
          data.price = parseFloat(priceMatch[1]);
          break;
        }
      }
    }

    // Extract brand
    const brandSelectors = [
      '#bylineInfo',
      '.a-link-normal[href*="brand"]',
      '[data-cy="byline-info"]',
      '.brand-link'
    ];

    for (const selector of brandSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        data.brand = element.textContent?.trim() || element.innerText?.trim();
        if (data.brand && data.brand.length > 0) break;
      }
    }

    // Extract category from breadcrumbs or title
    const breadcrumbSelectors = [
      '#wayfinding-breadcrumbs_feature_div .a-list-item',
      '.a-breadcrumb .a-link-normal',
      '#nav-subnav'
    ];

    for (const selector of breadcrumbSelectors) {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        const categories = Array.from(elements).map(el => el.textContent?.trim()).filter(Boolean);
        if (categories.length > 0) {
          data.category_code = categories.join('.').toLowerCase().replace(/\s+/g, '');
          break;
        }
      }
    }

    // Fallback: extract from title
    if (!data.category_code) {
      const title = document.querySelector('#productTitle')?.textContent?.trim();
      if (title) {
        // Simple category extraction from title keywords
        const categoryKeywords = ['electronics', 'smartphone', 'laptop', 'book', 'clothing', 'home', 'kitchen'];
        for (const keyword of categoryKeywords) {
          if (title.toLowerCase().includes(keyword)) {
            data.category_code = keyword;
            break;
          }
        }
      }
    }

    // Set defaults for missing data
    if (!data.price) data.price = 29.99; // Default price
    if (!data.brand) data.brand = 'unknown';
    if (!data.category_code) data.category_code = 'other';

    console.log('📊 Extracted Amazon data:', data);
    return data;

  } catch (error) {
    console.error('❌ Error extracting Amazon data:', error);
    return {
      price: 29.99,
      brand: 'unknown',
      category_code: 'other',
      hour: new Date().getHours(),
      day_of_week: new Date().getDay(),
      is_weekend: [0, 6].includes(new Date().getDay()) ? 1 : 0
    };
  }
}

// Update session data based on user behavior
function updateSessionData(action, value = 1) {
  sessionData.totalEvents += value;

  switch (action) {
    case 'scroll':
      sessionData.scrollEvents += value;
      break;
    case 'click':
      sessionData.clickEvents += value;
      break;
    case 'review_view':
      sessionData.reviewsViewed += value;
      break;
    case 'image_view':
      sessionData.imagesViewed += value;
      break;
    case 'add_to_cart':
      sessionData.addToCartClicked = true;
      break;
    case 'buy_now':
      sessionData.buyNowClicked = true;
      break;
    case 'share':
      sessionData.sharedProduct = true;
      break;
    case 'like':
      sessionData.likedProduct = true;
      break;
  }

  // Update time on page
  sessionData.timeOnPage = (Date.now() - sessionData.startTime) / 1000; // in seconds

  // Save to storage
  chrome.storage.local.set({ 'session_data': sessionData });

  console.log('📈 Updated session data:', sessionData);
}

// Calculate dynamic features based on realistic e-commerce funnel weights
function calculateDynamicFeatures(extractedData) {
  const timeOnPage = sessionData.timeOnPage;
  const totalEvents = sessionData.totalEvents;
  const reviewsViewed = sessionData.reviewsViewed;
  const imagesViewed = sessionData.imagesViewed;

  // Determine engagement level based on actions and behavior
  let engagementLevel = 'low';
  let funnelWeight = 0.05; // Default low engagement weight

  // Very High Engagement: Buy Now, Add to Cart, Share, Like (1-5% of users, 40-90% conversion)
  if (sessionData.buyNowClicked || sessionData.addToCartClicked) {
    engagementLevel = 'very_high';
    funnelWeight = 1.0;
  }
  // High Engagement: Multiple reviews + images + long time (3-10% of users, 10-25% conversion)
  else if ((reviewsViewed >= 3 || imagesViewed >= 2) && timeOnPage > 120) {
    engagementLevel = 'high';
    funnelWeight = 0.8;
  }
  // Medium Engagement: Some reviews or images + moderate time (10-25% of users, 1-3% conversion)
  else if ((reviewsViewed >= 1 || imagesViewed >= 1) && timeOnPage > 30) {
    engagementLevel = 'medium';
    funnelWeight = 0.4;
  }
  // Low Engagement: Minimal interaction (70-90% of users, 0.1-0.5% conversion)
  else if (totalEvents < 5 && timeOnPage < 30) {
    engagementLevel = 'low';
    funnelWeight = 0.05;
  }

  console.log(`🎯 Engagement Level: ${engagementLevel} (weight: ${funnelWeight})`);

  // Base features
  const basePrice = extractedData.price;
  const sessionDuration = Math.max(timeOnPage / 60, 0.1);

  // Apply very conservative funnel-based multipliers to prevent unrealistic predictions
  const engagementMultiplier = 1 + (funnelWeight * 0.2); // Much more conservative
  const intentMultiplier = funnelWeight * 1.2; // Reduced intent boost

  return {
    // Basic product features (unchanged - price is already a strong predictor)
    price: basePrice,
    brand: extractedData.brand,
    category_code: extractedData.category_code,

    // Temporal features (unchanged)
    hour: extractedData.hour,
    day_of_week: extractedData.day_of_week,
    is_weekend: extractedData.is_weekend,

    // Session features with very conservative scaling
    total_events_x: Math.max(totalEvents * engagementMultiplier, 1),
    unique_products: Math.max(1 + Math.floor(funnelWeight * 1), 1), // Reduced from 2
    unique_categories: Math.max(1 + Math.floor(reviewsViewed * 0.3), 1), // Reduced from 0.5
    avg_price: basePrice * (1 + (funnelWeight * 0.1)), // Much more conservative
    max_price: basePrice * (1 + (funnelWeight * 0.15)), // Much more conservative
    total_price_viewed: basePrice * (1 + (funnelWeight * 0.2)), // Much more conservative
    session_duration: sessionDuration * (1 + (funnelWeight * 0.15)), // Reduced

    // User history features with very conservative boosts
    avg_events_per_session: Math.max(totalEvents * engagementMultiplier, 1),
    total_events_y: Math.max(totalEvents * engagementMultiplier * 1.2, 1), // Reduced from 1.5
    avg_products_per_session: Math.max((1 + Math.floor(funnelWeight * 1)) * (1 + (reviewsViewed * 0.1)), 1), // Reduced
    total_products_viewed: Math.max(totalEvents * engagementMultiplier, 1),
    avg_session_duration: sessionDuration * (1 + (funnelWeight * 0.1)), // Reduced
    total_purchases: Math.min(Math.floor(intentMultiplier * 1.2), 2), // Capped at 2, reduced multiplier
    total_sessions: Math.max(1 + Math.floor(timeOnPage / 300) + Math.floor(funnelWeight * 2), 1), // Reduced from 3
    avg_price_viewed: basePrice * (1 + (funnelWeight * 0.15)), // More conservative
    conversion_rate: Math.min(funnelWeight * 0.4, 0.5) // Much more conservative, capped at 0.5
  };
}

// Enhanced prediction with dynamic behavior tracking
async function dynamicPrediction(force = false) {
  const now = Date.now();

  // Throttle predictions to avoid spam (max once per 2 seconds unless forced)
  if (!force && (now - sessionData.lastPredictionTime) < 2000) {
    return;
  }

  sessionData.lastPredictionTime = now;

  try {
    console.log('🔍 ML: Extracting data and calculating dynamic features...');

    // Extract data from current Amazon page
    const extractedData = extractAmazonProductData();

    // Calculate features based on user behavior
    const predictionData = calculateDynamicFeatures(extractedData);

    console.log('📤 Sending dynamic prediction data:', predictionData);

    const response = await fetch('http://127.0.0.1:8001/predict_event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(predictionData)
    });

    console.log('🔍 ML: Response status:', response.status);

    if (response.ok) {
      const data = await response.json();
      console.log('🔍 ML: Dynamic prediction received:', data);

      // Store in chrome storage for popup to read
      chrome.storage.local.set({
        'ml_prediction': data,
        'ml_prediction_timestamp': Date.now(),
        'extracted_product_data': extractedData,
        'session_data': sessionData
      }, () => {
        console.log('🔍 ML: Dynamic prediction and session data stored in chrome.storage');
      });

    } else {
      console.error('🔍 ML: API request failed:', response.status);
      const errorText = await response.text();
      console.error('🔍 ML: Error details:', errorText);
    }
  } catch (error) {
    console.error('🔍 ML: Error:', error);
  }
}

// Check if we're on Amazon
function isAmazonPage() {
  return window.location.hostname.includes('amazon.com') ||
         window.location.hostname.includes('amazon.');
}

// Initialize behavior tracking
function initializeBehaviorTracking() {
  console.log('🎯 Initializing behavior tracking...');

  // Track scrolling
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    updateSessionData('scroll', 0.1); // Fractional scroll events
    scrollTimeout = setTimeout(() => {
      dynamicPrediction(); // Update prediction after scroll stops
    }, 500);
  });

  // Track clicks
  document.addEventListener('click', (event) => {
    updateSessionData('click');

    // Specific action detection
    if (event.target.matches('#add-to-cart-button, #add-to-cart')) {
      updateSessionData('add_to_cart');
      console.log('🛒 Add to cart clicked!');
      dynamicPrediction(true); // Force immediate prediction
    } else if (event.target.matches('#buy-now-button, #buyNow')) {
      updateSessionData('buy_now');
      console.log('💰 Buy now clicked!');
      dynamicPrediction(true);
    } else if (event.target.matches('[data-action="share"], .share-button')) {
      updateSessionData('share');
      console.log('📤 Product shared!');
      dynamicPrediction(true);
    } else if (event.target.matches('.like-button, [data-action="like"]')) {
      updateSessionData('like');
      console.log('❤️ Product liked!');
      dynamicPrediction(true);
    }

    // Review/comment interaction
    if (event.target.closest('#reviews-summary, .review, .a-section.review')) {
      updateSessionData('review_view');
      console.log('📝 Review viewed!');
    }

    // Image viewing
    if (event.target.closest('[data-image-index], .image-thumbnail')) {
      updateSessionData('image_view');
      console.log('🖼️ Product image viewed!');
    }
  });

  // Track time on page
  setInterval(() => {
    updateSessionData('time_update');
    // Periodic prediction updates based on engagement
    if (sessionData.totalEvents > 5 && Math.random() < 0.3) { // 30% chance every 10 seconds if engaged
      dynamicPrediction();
    }
  }, 10000); // Every 10 seconds
}

// Initialize based on page type
if (isAmazonPage()) {
  console.log('🛒 Detected Amazon page, enabling dynamic behavior tracking');

  // Initialize tracking
  initializeBehaviorTracking();

  // Trigger initial prediction when page loads
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      setTimeout(() => dynamicPrediction(true), 2000); // Wait for dynamic content
    });
  } else {
    setTimeout(() => dynamicPrediction(true), 2000);
  }

  // Trigger on URL changes (for SPA navigation)
  let currentUrl = window.location.href;
  setInterval(() => {
    if (window.location.href !== currentUrl) {
      currentUrl = window.location.href;
      console.log('🔄 URL changed, resetting session and updating prediction...');
      // Reset session for new product
      sessionData = {
        startTime: Date.now(),
        totalEvents: 0,
        scrollEvents: 0,
        clickEvents: 0,
        timeOnPage: 0,
        reviewsViewed: 0,
        imagesViewed: 0,
        addToCartClicked: false,
        buyNowClicked: false,
        sharedProduct: false,
        likedProduct: false,
        lastPredictionTime: 0
      };
      setTimeout(() => dynamicPrediction(true), 1000);
    }
  }, 1000);

} else {
  console.log('ℹ️ Not on Amazon, using test prediction');
  // Fallback to test prediction for non-Amazon pages
  setTimeout(() => {
    console.log('🔍 ML: Triggering test prediction...');
    dynamicPrediction(true);
  }, 1000);
}

console.log('🔍 ML Content Script: Loaded successfully');
