document.getElementById("analyze").addEventListener("click", async () => {
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = '<div class="loading"><div class="loader"></div></div>';

  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    chrome.tabs.sendMessage(tab.id, { type: "GET_ECOMMERCE_ITEMS" }, (response) => {
      lastResponse = response;
      // New structure: { mainProduct, relatedProducts, bestBuy, items }
      if (response && (response.mainProduct || response.relatedProducts?.length > 0)) {
        let html = '';
        if (response.mainProduct) {
          html += `<h3>Main Product</h3><ul><li><strong>${response.mainProduct.name}</strong><br>Price: ${response.mainProduct.price}<br>Rating: ${response.mainProduct.rating || 'N/A'}<br>Specs:<ul>`;
          for (const [key, value] of Object.entries(response.mainProduct.specs || {})) {
            html += `<li>${key}: ${value}</li>`;
          }
          html += '</ul></li></ul>';
        }
        if (response.relatedProducts && response.relatedProducts.length > 0) {
          html += `<h3>Related Products</h3><ul>`;
          response.relatedProducts.forEach((item, idx) => {
            html += `<li><strong>${item.name}</strong><br>Price: ${item.price}<br>Rating: ${item.rating || 'N/A'}<br>Specs:<ul>`;
            for (const [key, value] of Object.entries(item.specs || {})) {
              html += `<li>${key}: ${value}</li>`;
            }
            html += '</ul></li>';
          });
          html += '</ul>';
        }
        if (response.bestBuy) {
          html += `<h3>Best Buy Recommendation</h3><div style="background:#e0ffe0;padding:8px;border-radius:6px;"><strong>${response.bestBuy.name}</strong><br>Price: ${response.bestBuy.price}<br>Rating: ${response.bestBuy.rating || 'N/A'}<br>Specs:<ul>`;
          for (const [key, value] of Object.entries(response.bestBuy.specs || {})) {
            html += `<li>${key}: ${value}</li>`;
          }
          html += '</ul></div>';
        }
        resultDiv.innerHTML = html;
        return;
      }
      // Fallback: show items array if present
      let products = response && response.items ? response.items : [];
      if (products.length > 0) {
        let html = '<ul>';
        products.forEach((item, idx) => {
          html += `<li><strong>${item.name}</strong><br>Price: ${item.price}<br>Rating: ${item.rating || 'N/A'}</li>`;
        });
        html += '</ul>';
        resultDiv.innerHTML = html;
        return;
      }
      // If no products found, use Gemini API
      chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_HTML" }, async (resp) => {
        const pageHtml = resp && resp.html ? resp.html : "";
        if (!pageHtml) {
          resultDiv.innerText = "Could not get page HTML.";
          return;
        }
        // Call Gemini API
        const geminiApiKey = "AIzaSyBzS7zo2QA0S6FsBA4xZGSaaAIUtEfoEDQ";
        const geminiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + geminiApiKey;
        const prompt = `Extract a list of products with their prices and ratings from this HTML. Return JSON array with fields: name, price, rating.\nHTML:\n${pageHtml}`;
        try {
          const geminiRes = await fetch(geminiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
          });
          const geminiData = await geminiRes.json();
          let text = "";
          if (geminiData && geminiData.candidates && geminiData.candidates[0].content && geminiData.candidates[0].content.parts && geminiData.candidates[0].content.parts[0].text) {
            text = geminiData.candidates[0].content.parts[0].text;
          }
          // Try to parse JSON from Gemini response
          let products = [];
          try {
            products = JSON.parse(text);
          } catch (e) {
            // If not valid JSON, show raw text
            resultDiv.innerText = text || "No products found.";
            return;
          }
          if (Array.isArray(products) && products.length > 0) {
            let html = '<ul>';
            products.forEach((item, idx) => {
              html += `<li><strong>${item.name}</strong><br>Price: ${item.price}<br>Rating: ${item.rating || 'N/A'}</li>`;
            });
            html += '</ul>';
            resultDiv.innerHTML = html;
          } else {
            resultDiv.innerText = "No products found.";
          }
        } catch (err) {
          resultDiv.innerText = "Gemini API error: " + err.message;
            }
          });
      });
    });
  });

let lastResponse = null;
let lastMLPrediction = null;

// Budget filter and best buy logic

document.getElementById("find-best-buy").addEventListener("click", () => {
  const resultDiv = document.getElementById("result");
  const minPrice = parseFloat(document.getElementById("min-price").value) || 0;
  const maxPrice = parseFloat(document.getElementById("max-price").value) || Number.MAX_SAFE_INTEGER;
  if (!lastResponse) {
    resultDiv.innerText = "Please analyze the page first.";
    return;
  }
  // Gather all products (main + related)
  let allProducts = [];
  if (lastResponse.mainProduct) allProducts.push(lastResponse.mainProduct);
  if (lastResponse.relatedProducts && lastResponse.relatedProducts.length > 0) {
    allProducts = allProducts.concat(lastResponse.relatedProducts);
  }
  // Filter by price range
  allProducts = allProducts.filter(p => {
    let priceNum = parseFloat((p.price || '').replace(/[^\d\.]/g, ''));
    return priceNum >= minPrice && priceNum <= maxPrice;
  });
  if (allProducts.length === 0) {
    resultDiv.innerHTML = `<div>No products found in this price range.</div>`;
    return;
  }
  // Sort by rating (desc), then price (asc)
  allProducts.sort((a, b) => {
    let ar = parseFloat((a.rating || '').match(/\d+(\.\d+)?/)?.[0] || '0');
    let br = parseFloat((b.rating || '').match(/\d+(\.\d+)?/)?.[0] || '0');
    if (br !== ar) return br - ar;
    let ap = parseFloat((a.price || '').replace(/[^\d\.]/g, ''));
    let bp = parseFloat((b.price || '').replace(/[^\d\.]/g, ''));
    return ap - bp;
  });
  let bestBuy = allProducts[0];
  let html = `<h3>Best Buy in Your Budget</h3><div style="background:#e0ffe0;padding:8px;border-radius:6px;"><strong>${bestBuy.name}</strong><br>Price: ${bestBuy.price}<br>Rating: ${bestBuy.rating || 'N/A'}<br>Specs:<ul>`;
  for (const [key, value] of Object.entries(bestBuy.specs || {})) {
    html += `<li>${key}: ${value}</li>`;
  }
  html += '</ul></div>';
  html += `<h3>Other Products in Range</h3><ul>`;
  allProducts.slice(1).forEach((item, idx) => {
    html += `<li><strong>${item.name}</strong><br>Price: ${item.price}<br>Rating: ${item.rating || 'N/A'}<br>Specs:<ul>`;
    for (const [key, value] of Object.entries(item.specs || {})) {
      html += `<li>${key}: ${value}</li>`;
    }
    html += '</ul></li>';
  });
  html += '</ul>';
  resultDiv.innerHTML = html;
});

// Copy button functionality
document.getElementById("copy-btn").addEventListener("click", () => {
  const resultDiv = document.getElementById("result");
  const text = resultDiv.innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("Results copied to clipboard!");
  });
});

// Load ML prediction from storage when popup opens
chrome.storage.local.get(['ml_prediction', 'ml_prediction_timestamp'], (result) => {
  console.log('🔍 Popup: Loading ML prediction from storage:', result);
  if (result.ml_prediction) {
    console.log('🔍 Popup: Found ML prediction, updating UI');
    lastMLPrediction = result.ml_prediction;
    updateMLPredictionUI();
  } else {
    console.log('🔍 Popup: No ML prediction found in storage');
  }
});

function updateMLPredictionUI() {
  const mlDiv = document.getElementById('ml-prediction');
  if (!mlDiv) {
    // Create container if not exists
    const container = document.createElement('div');
    container.id = 'ml-prediction';
    container.style.padding = '10px';
    container.style.marginTop = '10px';
    container.style.backgroundColor = '#f0f8ff';
    container.style.borderRadius = '6px';
    container.style.border = '1px solid #007bff';
    container.style.fontSize = '14px';
    container.style.fontWeight = 'bold';
    const resultDiv = document.getElementById('result');
    resultDiv.parentNode.insertBefore(container, resultDiv.nextSibling);
  }
  const displayDiv = document.getElementById('ml-prediction');
  if (lastMLPrediction) {
    displayDiv.innerHTML = `
      Purchase Probability: ${(lastMLPrediction.purchase_probability * 100).toFixed(2)}%<br>
      Recommendation: <span style="color:${getRecommendationColor(lastMLPrediction.recommendation)}">${lastMLPrediction.recommendation.toUpperCase()}</span><br>
      Features Used: ${lastMLPrediction.feature_count}
    `;
  } else {
    displayDiv.innerHTML = 'No ML prediction available yet.';
  }
}

function getRecommendationColor(rec) {
  if (!rec) return '#808080'; // gray for null/undefined
  switch (rec.toLowerCase()) {
    case 'very_high': return '#ff0000'; // red
    case 'high': return '#ff4500'; // orange red
    case 'medium': return '#ffa500'; // orange
    case 'low': return '#32cd32'; // lime green
    case 'very_low': return '#008000'; // green
    default: return '#000000'; // black
  }
}

// Request latest ML prediction on popup open
document.addEventListener('DOMContentLoaded', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length > 0) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_ML_PREDICTION' }, (response) => {
        if (response && response.prediction) {
          lastMLPrediction = response.prediction;
          updateMLPredictionUI();
        }
      });
    }
  });
});

// -----------------------
// Amazon Sentiment Analysis
// -----------------------

function localLexiconSentimentBatch(texts) {
  const positiveWords = [
    "good","great","awesome","amazing","love","excellent","nice","helpful","cool","fantastic","best","well done","brilliant","super","wow","thanks","thank you","useful","liked","enjoyed","informative","clear","perfect","satisfied","happy","wonderful","outstanding","incredible","recommend","worth","quality","fast","quick","reliable","comfortable","beautiful","solid","sturdy","durable","pleased","impressed","smooth","easy","convenient","value","affordable","deal","bargain","effective","works","working","fine","decent","fair","reasonable"
  ];
  const negativeWords = [
    "bad","terrible","awful","hate","worst","boring","useless","waste","trash","stupid","dumb","confusing","poor","dislike","bug","broken","lag","slow","misleading","clickbait","annoying","cringe","horrible","disgusting","disappointing","frustrated","angry","mad","upset","sad","unhappy","dissatisfied","regret","mistake","wrong","error","problem","issue","trouble","difficulty","hard","tough","complicated","cheap","expensive","overpriced","fake","counterfeit","defective","damaged","scratched","worn","old","outdated","useless","worthless","junk","garbage","failed","failure","disaster"
  ];
  
  const toScore = (text) => {
    const t = (text || "").toLowerCase();
    let pos = 0, neg = 0;
    for (const w of positiveWords) if (t.includes(w)) pos++;
    for (const w of negativeWords) if (t.includes(w)) neg++;
    const total = pos + neg;
    if (total === 0) return { label: "neutral", score: 0.5 };
    const confidence = Math.min(1, Math.abs(pos - neg) / Math.max(1, total));
    if (pos > neg) return { label: "positive", score: Number((0.6 + 0.4 * confidence).toFixed(4)) };
    if (neg > pos) return { label: "negative", score: Number((0.6 + 0.4 * confidence).toFixed(4)) };
    return { label: "neutral", score: 0.5 };
  };
  
  return texts.map(toScore);
}

async function analyzeAmazonReviews() {
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = '<div class="loading"><div class="loader"></div></div>';

  chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
    if (!tab.url.includes('amazon')) {
      resultDiv.innerText = "This feature only works on Amazon product pages. Please navigate to an Amazon product page first.";
      return;
    }

    chrome.tabs.sendMessage(tab.id, { type: "GET_AMAZON_REVIEWS" }, (response) => {
      if (!response) {
        resultDiv.innerText = "Unable to extract reviews. Please make sure you're on an Amazon product page with visible reviews.";
        return;
      }

      if (response.error) {
        let errorMsg = response.error;
        if (response.reviewsLink) {
          errorMsg += `\n\nClick here to go to reviews: ${response.reviewsLink}`;
        }
        resultDiv.innerText = errorMsg;
        return;
      }

      const reviews = response.reviews || [];
      if (reviews.length === 0) {
        resultDiv.innerText = "No reviews found on this page. Try navigating to the product reviews section.";
        return;
      }

      try {
        // Perform sentiment analysis
        const reviewTexts = reviews.map(r => r.text);
        const sentimentResults = localLexiconSentimentBatch(reviewTexts);
        
        const analyzedReviews = reviews.map((review, index) => ({
          ...review,
          sentiment: sentimentResults[index].label,
          confidence: sentimentResults[index].score
        }));

        // Calculate summary statistics
        const counts = { positive: 0, neutral: 0, negative: 0 };
        let totalConfidence = 0;
        
        analyzedReviews.forEach(review => {
          counts[review.sentiment]++;
          totalConfidence += review.confidence;
        });

        const avgConfidence = analyzedReviews.length ? totalConfidence / analyzedReviews.length : 0;

        // Render results
        let html = `<h3>Amazon Reviews Sentiment Analysis</h3>`;
        html += `<div style="background:#f0f8ff;padding:10px;border-radius:6px;margin-bottom:15px;">`;
        html += `<strong>Summary:</strong><br>`;
        html += `Total Reviews Analyzed: ${analyzedReviews.length}<br>`;
        html += `Positive: ${counts.positive} (${(counts.positive/analyzedReviews.length*100).toFixed(1)}%)<br>`;
        html += `Neutral: ${counts.neutral} (${(counts.neutral/analyzedReviews.length*100).toFixed(1)}%)<br>`;
        html += `Negative: ${counts.negative} (${(counts.negative/analyzedReviews.length*100).toFixed(1)}%)<br>`;
        html += `Average Confidence: ${avgConfidence.toFixed(3)}<br>`;
        html += `</div>`;

        html += `<h4>Review Details (showing first 15):</h4>`;
        
        analyzedReviews.slice(0, 15).forEach((review, index) => {
          const bgColor = review.sentiment === 'positive' ? '#e8f5e8' : 
                         review.sentiment === 'negative' ? '#ffe8e8' : '#f5f5f5';
          
          html += `<div style="background:${bgColor};padding:8px;margin:5px 0;border-radius:4px;border-left:4px solid ${review.sentiment === 'positive' ? '#4CAF50' : review.sentiment === 'negative' ? '#f44336' : '#9e9e9e'};">`;
          html += `<strong>${review.sentiment.toUpperCase()}</strong> (${review.confidence.toFixed(2)})<br>`;
          if (review.rating) html += `Rating: ${review.rating}<br>`;
          if (review.reviewer) html += `Reviewer: ${review.reviewer}<br>`;
          html += `<em>"${review.text.substring(0, 200)}${review.text.length > 200 ? '...' : ''}"</em>`;
          html += `</div>`;
        });

        if (analyzedReviews.length > 15) {
          html += `<p><em>...and ${analyzedReviews.length - 15} more reviews</em></p>`;
        }

        resultDiv.innerHTML = html;

      } catch (error) {
        resultDiv.innerText = `Error analyzing reviews: ${error.message}`;
      }
    });
  });
}

// Add event listener for the sentiment analysis button
document.getElementById("analyze-sentiment").addEventListener("click", analyzeAmazonReviews);

// -----------------------
// Product Comparison Feature
// -----------------------

// Function to correct typos and normalize product names using Gemini
async function normalizeProductName(productName) {
  const geminiApiKey = "AIzaSyBzS7zo2QA0S6FsBA4xZGSaaAIUtEfoEDQ";
  const geminiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + geminiApiKey;
  
  const prompt = `Please correct any typos and normalize this product name for comparison: "${productName}". 
  Return only the corrected product name, nothing else. If it's already correct, return it as is.
  Examples:
  - "iphone 15 pro" -> "iPhone 15 Pro"
  - "samsng galxy s24" -> "Samsung Galaxy S24"
  - "macbok air m2" -> "MacBook Air M2"`;

  try {
    const response = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
    });
    
    const data = await response.json();
    if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
      return data.candidates[0].content.parts[0].text.trim();
    }
    return productName; // Fallback to original if API fails
  } catch (error) {
    console.error("Error normalizing product name:", error);
    return productName; // Fallback to original if API fails
  }
}

// Function to search for product information using Gemini
async function searchProductInfo(productName) {
  const geminiApiKey = "AIzaSyBzS7zo2QA0S6FsBA4xZGSaaAIUtEfoEDQ";
  const geminiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + geminiApiKey;
  
  // Try multiple prompt strategies
  const prompts = [
    // Strategy 1: Direct JSON request
    `Product: ${productName}
Return only this JSON format:
{"name":"${productName}","price":"$XXX","rating":"X.X/5","specifications":{"Brand":"","Display":"","Processor":"","RAM":"","Storage":"","Camera":"","Battery":"","OS":""},"pros":["","",""],"cons":["","",""]}`,
    
    // Strategy 2: Simple request
    `What is the price, rating, and key specifications of ${productName}? Return as JSON only.`,
    
    // Strategy 3: Specific details request
    `${productName} smartphone details: price range, display size, processor, RAM, storage, camera, battery. JSON format only.`
  ];

  // Try each strategy
  for (let i = 0; i < prompts.length; i++) {
    try {
      console.log(`Trying strategy ${i + 1} for ${productName}`);
      
      const response = await fetch(geminiUrl, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ 
          contents: [{ 
            parts: [{ text: prompts[i] }] 
          }] 
        })
      });
      
      console.log(`Strategy ${i + 1} - API Response status: ${response.status}`);
      
      if (!response.ok) {
        console.error(`Strategy ${i + 1} failed with status: ${response.status}`);
        continue; // Try next strategy
      }
      
      const data = await response.json();
      
      if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
        const responseText = data.candidates[0].content.parts[0].text.trim();
        console.log(`Strategy ${i + 1} response:`, responseText);
        
        const parsedData = parseProductResponse(responseText, productName);
        if (parsedData && parsedData.price !== "Price information unavailable") {
          console.log(`Strategy ${i + 1} successful:`, parsedData);
          return parsedData;
        }
      }
      
      // Small delay between attempts
      await new Promise(resolve => setTimeout(resolve, 500));
      
    } catch (error) {
      console.error(`Strategy ${i + 1} error:`, error);
      continue; // Try next strategy
    }
  }
  
  // If all strategies fail, try to get basic info from product name
  console.log(`All API strategies failed for ${productName}, using enhanced fallback`);
  return createEnhancedFallbackInfo(productName);
}

// Enhanced function to parse product response with multiple methods
function parseProductResponse(responseText, productName) {
  try {
    // Method 1: Direct JSON parsing
    let cleanJson = responseText.trim();
    
    // Remove markdown formatting
    cleanJson = cleanJson.replace(/```json\s*|\s*```/g, '');
    cleanJson = cleanJson.replace(/```\s*|\s*```/g, '');
    
    // Try to find JSON in the response
    const jsonMatch = cleanJson.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      cleanJson = jsonMatch[0];
    }
    
    try {
      return JSON.parse(cleanJson);
    } catch (e) {
      console.log("Direct JSON parse failed, trying text extraction");
    }
    
    // Method 2: Extract information from text
    return extractInfoFromText(responseText, productName);
    
  } catch (error) {
    console.error("Error parsing response:", error);
    return null;
  }
}

// Function to extract product info from free-form text
function extractInfoFromText(text, productName) {
  const lowerText = text.toLowerCase();
  const result = {
    name: productName,
    price: "Price not found",
    rating: "Rating not found",
    specifications: {},
    pros: [],
    cons: []
  };
  
  // Extract price
  const pricePatterns = [
    /\$[\d,]+(?:\.\d{2})?/g,
    /₹[\d,]+(?:\.\d{2})?/g,
    /price[:\s]*\$?[\d,]+/gi,
    /cost[:\s]*\$?[\d,]+/gi,
    /around \$?[\d,]+/gi
  ];
  
  for (const pattern of pricePatterns) {
    const priceMatch = text.match(pattern);
    if (priceMatch) {
      result.price = priceMatch[0];
      break;
    }
  }
  
  // Extract rating
  const ratingPatterns = [
    /(\d+\.?\d*)\s*\/\s*5/g,
    /(\d+\.?\d*)\s*out\s*of\s*5/gi,
    /rating[:\s]*(\d+\.?\d*)/gi
  ];
  
  for (const pattern of ratingPatterns) {
    const ratingMatch = text.match(pattern);
    if (ratingMatch) {
      result.rating = `${ratingMatch[1]}/5`;
      break;
    }
  }
  
  // Extract specifications
  const specPatterns = {
    "Display": /display[:\s]*([^\n\r,]+)/gi,
    "Processor": /processor[:\s]*([^\n\r,]+)/gi,
    "RAM": /ram[:\s]*([^\n\r,]+)/gi,
    "Storage": /storage[:\s]*([^\n\r,]+)/gi,
    "Camera": /camera[:\s]*([^\n\r,]+)/gi,
    "Battery": /battery[:\s]*([^\n\r,]+)/gi
  };
  
  for (const [spec, pattern] of Object.entries(specPatterns)) {
    const match = text.match(pattern);
    if (match && match[1]) {
      result.specifications[spec] = match[1].trim();
    }
  }
  
  return result;
}

// Enhanced fallback with real product data when possible
function createEnhancedFallbackInfo(productName) {
  console.log(`Creating enhanced fallback for ${productName}`);
  
  const lowerName = productName.toLowerCase();
  let result = {
    name: productName,
    price: "Price information unavailable",
    rating: "Rating information unavailable",
    specifications: {},
    pros: [],
    cons: []
  };
  
  // Enhanced Vivo product information
  if (lowerName.includes('vivo t2')) {
    result = {
      name: "Vivo T2 5G",
      price: "$200 - $300 (approx)",
      rating: "4.1/5 (estimated)",
      specifications: {
        "Brand": "Vivo",
        "Display": "6.38-inch AMOLED, 90Hz",
        "Processor": "Snapdragon 695 5G",
        "RAM": "6GB/8GB",
        "Storage": "128GB/256GB",
        "Camera": "64MP Triple Camera",
        "Battery": "4500mAh, 44W Fast Charging",
        "OS": "Android 13, Funtouch OS 13"
      },
      pros: ["Good performance for price", "Decent camera quality", "Fast charging", "5G connectivity"],
      cons: ["Plastic build", "No wireless charging", "Limited availability", "Average low-light camera performance"]
    };
  } else if (lowerName.includes('vivo t4')) {
    result = {
      name: "Vivo T4 5G",
      price: "$250 - $350 (approx)",
      rating: "4.3/5 (estimated)",
      specifications: {
        "Brand": "Vivo",
        "Display": "6.62-inch AMOLED, 120Hz",
        "Processor": "Snapdragon 7 Gen 3",
        "RAM": "8GB/12GB",
        "Storage": "128GB/256GB",
        "Camera": "50MP OIS Triple Camera",
        "Battery": "5000mAh, 80W Fast Charging",
        "OS": "Android 14, Funtouch OS 14"
      },
      pros: ["Improved performance", "Better camera with OIS", "Faster charging", "Higher refresh rate display"],
      cons: ["Higher price than T2", "Still plastic build", "Limited global availability", "Bloatware in software"]
    };
  } else if (lowerName.includes('iphone 15')) {
    result = {
      name: "iPhone 15",
      price: "$799 - $899",
      rating: "4.6/5",
      specifications: {
        "Brand": "Apple",
        "Display": "6.1-inch Super Retina XDR",
        "Processor": "A16 Bionic",
        "RAM": "6GB",
        "Storage": "128GB/256GB/512GB",
        "Camera": "48MP Main + 12MP Ultra Wide",
        "Battery": "Up to 20 hours video playback",
        "OS": "iOS 17"
      },
      pros: ["Excellent performance", "Great camera quality", "Premium build", "Long software support"],
      cons: ["Expensive", "No USB-C", "Limited customization", "No high refresh rate"]
    };
  } else if (lowerName.includes('samsung') && lowerName.includes('s24')) {
    result = {
      name: "Samsung Galaxy S24",
      price: "$799 - $999",
      rating: "4.5/5",
      specifications: {
        "Brand": "Samsung",
        "Display": "6.2-inch Dynamic AMOLED 2X, 120Hz",
        "Processor": "Snapdragon 8 Gen 3",
        "RAM": "8GB",
        "Storage": "128GB/256GB/512GB",
        "Camera": "50MP Triple Camera System",
        "Battery": "4000mAh, 25W Fast Charging",
        "OS": "Android 14, One UI 6.1"
      },
      pros: ["Excellent display", "Powerful performance", "Versatile cameras", "AI features"],
      cons: ["Smaller battery", "Expensive", "No charger included", "Bloatware"]
    };
  }
  
  // Add note about data source
  result.specifications["Note"] = "Estimated specifications - exact details may vary";
  
  return result;
}

// Function to render comparison table
function renderComparisonTable(product1, product2) {
  if (!product1 || !product2) {
    return '<div>Error: Could not retrieve product information. Please check your internet connection and try again.</div>';
  }

  let html = '<h3>Product Comparison</h3>';
  html += '<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;">';
  
  // Header row
  html += '<tr style="background:#f5f5f5;">';
  html += '<th style="border:1px solid #ddd;padding:8px;text-align:left;">Feature</th>';
  html += `<th style="border:1px solid #ddd;padding:8px;text-align:center;">${product1.name}</th>`;
  html += `<th style="border:1px solid #ddd;padding:8px;text-align:center;">${product2.name}</th>`;
  html += '</tr>';

  // Price row
  html += '<tr>';
  html += '<td style="border:1px solid #ddd;padding:8px;font-weight:bold;">Price</td>';
  html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product1.price}</td>`;
  html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product2.price}</td>`;
  html += '</tr>';

  // Rating row
  html += '<tr style="background:#f9f9f9;">';
  html += '<td style="border:1px solid #ddd;padding:8px;font-weight:bold;">Rating</td>';
  html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product1.rating}</td>`;
  html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product2.rating}</td>`;
  html += '</tr>';

  // Specifications
  const allSpecs = new Set([
    ...Object.keys(product1.specifications || {}),
    ...Object.keys(product2.specifications || {})
  ]);

  let isAlternate = false;
  allSpecs.forEach(spec => {
    html += `<tr${isAlternate ? ' style="background:#f9f9f9;"' : ''}>`;
    html += `<td style="border:1px solid #ddd;padding:8px;font-weight:bold;">${spec}</td>`;
    html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product1.specifications?.[spec] || 'N/A'}</td>`;
    html += `<td style="border:1px solid #ddd;padding:8px;text-align:center;">${product2.specifications?.[spec] || 'N/A'}</td>`;
    html += '</tr>';
    isAlternate = !isAlternate;
  });

  html += '</table>';

  // Pros and Cons
  html += '<div style="display:flex;gap:20px;margin-top:20px;">';
  
  // Product 1 Pros/Cons
  html += '<div style="flex:1;">';
  html += `<h4>${product1.name}</h4>`;
  if (product1.pros && product1.pros.length > 0) {
    html += '<div style="background:#e8f5e8;padding:8px;border-radius:4px;margin-bottom:8px;">';
    html += '<strong>Pros:</strong><ul>';
    product1.pros.forEach(pro => html += `<li>${pro}</li>`);
    html += '</ul></div>';
  }
  if (product1.cons && product1.cons.length > 0) {
    html += '<div style="background:#ffe8e8;padding:8px;border-radius:4px;">';
    html += '<strong>Cons:</strong><ul>';
    product1.cons.forEach(con => html += `<li>${con}</li>`);
    html += '</ul></div>';
  }
  html += '</div>';

  // Product 2 Pros/Cons
  html += '<div style="flex:1;">';
  html += `<h4>${product2.name}</h4>`;
  if (product2.pros && product2.pros.length > 0) {
    html += '<div style="background:#e8f5e8;padding:8px;border-radius:4px;margin-bottom:8px;">';
    html += '<strong>Pros:</strong><ul>';
    product2.pros.forEach(pro => html += `<li>${pro}</li>`);
    html += '</ul></div>';
  }
  if (product2.cons && product2.cons.length > 0) {
    html += '<div style="background:#ffe8e8;padding:8px;border-radius:4px;">';
    html += '<strong>Cons:</strong><ul>';
    product2.cons.forEach(con => html += `<li>${con}</li>`);
    html += '</ul></div>';
  }
  html += '</div>';

  html += '</div>';

  return html;
}

// Main comparison function
async function compareProducts() {
  const resultDiv = document.getElementById("result");
  const product1Name = document.getElementById("product1").value.trim();
  const product2Name = document.getElementById("product2").value.trim();

  if (!product1Name || !product2Name) {
    resultDiv.innerHTML = '<div style="color:red;">Please enter both product names to compare.</div>';
    return;
  }

  resultDiv.innerHTML = '<div class="loading"><div class="loader"></div><p>Searching and comparing products...</p></div>';

  try {
    console.log(`Starting comparison: ${product1Name} vs ${product2Name}`);
    
    // Step 1: Normalize product names (handle typos)
    console.log("Step 1: Normalizing product names...");
    const normalizedProduct1 = await normalizeProductName(product1Name);
    const normalizedProduct2 = await normalizeProductName(product2Name);
    console.log(`Normalized: ${normalizedProduct1} vs ${normalizedProduct2}`);

    // Step 2: Search for product information
    console.log("Step 2: Searching for product information...");
    resultDiv.innerHTML = '<div class="loading"><div class="loader"></div><p>Fetching product details...</p></div>';
    
    const productInfo1 = await searchProductInfo(normalizedProduct1);
    console.log("Product 1 info received:", productInfo1);
    
    resultDiv.innerHTML = '<div class="loading"><div class="loader"></div><p>Fetching second product details...</p></div>';
    
    const productInfo2 = await searchProductInfo(normalizedProduct2);
    console.log("Product 2 info received:", productInfo2);

    // Step 3: Render comparison table
    console.log("Step 3: Rendering comparison table...");
    
    if (!productInfo1 || !productInfo2) {
      resultDiv.innerHTML = '<div style="color:red;">Error: Could not retrieve product information. Please check the console for details and try again.</div>';
      return;
    }
    
    const comparisonHtml = renderComparisonTable(productInfo1, productInfo2);
    resultDiv.innerHTML = comparisonHtml;
    
    console.log("Comparison completed successfully");

  } catch (error) {
    console.error("Error in compareProducts:", error);
    resultDiv.innerHTML = `<div style="color:red;">
      <h4>Comparison Error</h4>
      <p>Error: ${error.message}</p>
      <p>Please try the following:</p>
      <ul>
        <li>Check your internet connection</li>
        <li>Try more specific product names</li>
        <li>Wait a moment and try again</li>
        <li>Check the browser console for detailed error information</li>
      </ul>
    </div>`;
  }
}

// Add event listener for the compare products button
document.getElementById("compare-products").addEventListener("click", compareProducts);

// Add event listener for enhanced comparison (bypasses API)
document.getElementById("compare-enhanced").addEventListener("click", async function() {
  const resultDiv = document.getElementById("result");
  const product1Name = document.getElementById("product1").value.trim();
  const product2Name = document.getElementById("product2").value.trim();

  if (!product1Name || !product2Name) {
    resultDiv.innerHTML = '<div style="color:red;">Please enter both product names to compare.</div>';
    return;
  }

  resultDiv.innerHTML = '<div class="loading"><div class="loader"></div><p>Loading enhanced product data...</p></div>';

  try {
    // Use enhanced fallback data directly (no API calls)
    const productInfo1 = createEnhancedFallbackInfo(product1Name);
    const productInfo2 = createEnhancedFallbackInfo(product2Name);

    const comparisonHtml = renderComparisonTable(productInfo1, productInfo2);
    resultDiv.innerHTML = comparisonHtml;

  } catch (error) {
    resultDiv.innerHTML = `<div style="color:red;">Error: ${error.message}</div>`;
  }
});

// Add keyboard support for comparison (Enter key)
document.getElementById("product1").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    compareProducts();
  }
});

document.getElementById("product2").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    compareProducts();
  }
});

// Add event listener for the clear comparison button
document.getElementById("clear-comparison").addEventListener("click", function() {
  document.getElementById("product1").value = "";
  document.getElementById("product2").value = "";
  document.getElementById("result").innerHTML = "Click 'Analyze E-commerce Page' to list all products, prices, and ratings found on this page.";
});

// Test API function for debugging
async function testAPI() {
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = '<div class="loading"><div class="loader"></div><p>Testing API connection...</p></div>';
  
  try {
    const geminiApiKey = "AIzaSyBzS7zo2QA0S6FsBA4xZGSaaAIUtEfoEDQ";
    const geminiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + geminiApiKey;
    
    const testPrompt = `Return this exact JSON: {"test": "success", "message": "API is working"}`;
    
    console.log("Testing API with URL:", geminiUrl);
    
    const response = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: testPrompt }] }] })
    });
    
    console.log("Test API Response status:", response.status);
    
    const data = await response.json();
    console.log("Test API Response data:", data);
    
    let html = '<h3>API Test Results</h3>';
    html += `<div style="background:#f0f8ff;padding:10px;border-radius:6px;">`;
    html += `<strong>Status:</strong> ${response.status}<br>`;
    html += `<strong>Response:</strong> ${response.ok ? 'SUCCESS' : 'FAILED'}<br>`;
    
    if (data?.candidates?.[0]?.content?.parts?.[0]?.text) {
      html += `<strong>API Response:</strong> ${data.candidates[0].content.parts[0].text}<br>`;
      html += `<div style="color:green;">✅ API is working correctly!</div>`;
    } else {
      html += `<div style="color:red;">❌ API response format unexpected</div>`;
      html += `<strong>Full Response:</strong> ${JSON.stringify(data)}<br>`;
    }
    
    html += `</div>`;
    resultDiv.innerHTML = html;
    
  } catch (error) {
    console.error("API Test Error:", error);
    let html = '<h3>API Test Results</h3>';
    html += `<div style="background:#ffe8e8;padding:10px;border-radius:6px;">`;
    html += `<div style="color:red;">❌ API Test Failed</div>`;
    html += `<strong>Error:</strong> ${error.message}<br>`;
    html += `<strong>Possible causes:</strong><br>`;
    html += `- Network connectivity issues<br>`;
    html += `- API key quota exceeded<br>`;
    html += `- CORS policy restrictions<br>`;
    html += `- API service temporarily unavailable<br>`;
    html += `</div>`;
    resultDiv.innerHTML = html;
  }
}

// Add event listener for the test API button
document.getElementById("test-api").addEventListener("click", testAPI);


