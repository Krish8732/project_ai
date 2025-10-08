
function getAllEcommerceItems() {
  let items = [];
  let mainProduct = null;
  let relatedProducts = [];

  // Amazon product page (main product)
  if (window.location.hostname.includes('amazon')) {
    let name = document.getElementById('productTitle')?.innerText?.trim() || "";
    let price = document.getElementById('priceblock_ourprice')?.innerText?.trim() ||
                document.getElementById('priceblock_dealprice')?.innerText?.trim() ||
                document.querySelector('.a-price .a-offscreen')?.innerText?.trim() || "";
    let rating = document.querySelector('.a-icon-star span')?.innerText?.trim() ||
                 document.getElementById('acrPopover')?.getAttribute('title') || "";
    // Specs table
    let specs = {};
    let specsTable = document.getElementById('productDetails_techSpec_section_1') || document.getElementById('productDetails_detailBullets_sections1');
    if (specsTable) {
      specsTable.querySelectorAll('tr').forEach(row => {
        let key = row.querySelector('th, td')?.innerText?.trim();
        let value = row.querySelector('td:last-child')?.innerText?.trim();
        if (key && value) specs[key] = value;
      });
    } else {
      // Try bullet points
      document.querySelectorAll('#feature-bullets ul li span').forEach((el, idx) => {
        specs[`Feature ${idx+1}`] = el.innerText.trim();
      });
    }
    if (name && price) {
      mainProduct = { name, price, rating, specs };
    }

    // Related products ("Sponsored", "Similar items", etc.)
    let relatedSelectors = [
      '#sp_detail', // Sponsored
      '#sims-consolidated-2_feature_div', // Similar items
      '#purchase-sims-feature', // Customers also bought
      '.a-carousel-card', // Carousel cards
      '.p13n-sc-uncoverable-faceout', // Best sellers
      '.a-section.a-spacing-none.p13n-asin' // More best sellers
    ];
    let relatedContainers = [];
    for (const sel of relatedSelectors) {
      let found = document.querySelectorAll(sel);
      if (found.length) relatedContainers = relatedContainers.concat(Array.from(found));
    }
    relatedContainers.forEach(card => {
      let rName = card.querySelector('img[alt]')?.getAttribute('alt') || card.querySelector('.a-link-normal')?.innerText?.trim() || "";
      let rPrice = card.querySelector('.a-price .a-offscreen')?.innerText?.trim() || card.querySelector('.a-price-whole')?.innerText?.trim() || "";
      let rRating = card.querySelector('.a-icon-alt')?.innerText?.trim() || "";
      let rSpecs = {};
      // Try to extract specs from tooltip or description
      let desc = card.querySelector('.a-row.a-size-small')?.innerText?.trim() || "";
      if (desc) rSpecs['Description'] = desc;
      if (rName && rPrice) {
        relatedProducts.push({ name: rName, price: rPrice, rating: rRating, specs: rSpecs });
      }
    });
  }

  // If not Amazon, fallback to generic extraction
  if (!mainProduct && items.length === 0) {
    const selectors = [
      '.product', '.product-item', '.product-card', '[itemtype*="Product"]', '.product-list-item'
    ];
    let containers = [];
    for (const sel of selectors) {
      containers = document.querySelectorAll(sel);
      if (containers.length) break;
    }
    if (!containers.length) containers = document.querySelectorAll('li, div'); // fallback

    containers.forEach(container => {
      let name = container.querySelector('[itemprop="name"], .product-title, .productName, h2, h3, .title')?.innerText?.trim() || "";
      let price = container.querySelector('[itemprop="price"], .price, .product-price, .price-tag, .a-price-whole')?.innerText?.trim() || "";
      let rating = container.querySelector('[itemprop="ratingValue"], .rating, .star-rating, .review-rating, .a-icon-alt')?.innerText?.trim() || "";
      let specs = {};
      if (!name && container.innerText) {
        const priceMatch = container.innerText.match(/\$\s?\d+[\.,]?\d*/);
        if (priceMatch) price = priceMatch[0];
        const ratingMatch = container.innerText.match(/\d+(\.\d+)?\s*out of\s*5/);
        if (ratingMatch) rating = ratingMatch[0];
        name = container.innerText.split('\n')[0].trim();
      }
      if (name && price) {
        items.push({ name, price, rating, specs });
      }
    });
  }

  // Compare and suggest best buy
  let bestBuy = null;
  if (mainProduct && relatedProducts.length > 0) {
    // Simple comparison: highest rating, then lowest price
    let allProducts = [mainProduct, ...relatedProducts];
    allProducts = allProducts.filter(p => p.price && p.rating);
    allProducts.sort((a, b) => {
      // Extract numeric rating
      let ar = parseFloat((a.rating || '').match(/\d+(\.\d+)?/)?.[0] || '0');
      let br = parseFloat((b.rating || '').match(/\d+(\.\d+)?/)?.[0] || '0');
      if (br !== ar) return br - ar;
      // Extract numeric price
      let ap = parseFloat((a.price || '').replace(/[^\d\.]/g, ''));
      let bp = parseFloat((b.price || '').replace(/[^\d\.]/g, ''));
      return ap - bp;
    });
    bestBuy = allProducts[0];
  }

  // Fallback: demo products if nothing found
  if (!mainProduct && items.length === 0) {
    items = [
      { name: "Demo Product 1", price: "$99.99", rating: "4.5 out of 5", specs: {} },
      { name: "Demo Product 2", price: "$49.99", rating: "4.0 out of 5", specs: {} },
      { name: "Demo Product 3", price: "$19.99", rating: "3.5 out of 5", specs: {} }
    ];
  }

  return { mainProduct, relatedProducts, bestBuy, items };
}




function getAmazonReviews() {
  let reviews = [];
  
  // Check if we're on Amazon
  if (!window.location.hostname.includes('amazon')) {
    return { error: "This feature only works on Amazon product pages" };
  }

  // Try different Amazon review selectors
  const reviewSelectors = [
    '[data-hook="review-body"] span',
    '.review-text',
    '.cr-original-review-text',
    '[data-hook="review-body"] > span > span',
    '.a-size-base.review-text.review-text-content',
    '[data-hook="review-body"] span:not(.cr-original-review-text)'
  ];

  let reviewElements = [];
  for (const selector of reviewSelectors) {
    reviewElements = document.querySelectorAll(selector);
    if (reviewElements.length > 0) break;
  }

  // If no reviews found on current page, try to navigate to reviews section
  if (reviewElements.length === 0) {
    // Look for "See all reviews" link or reviews section
    const reviewsLink = document.querySelector('[data-hook="see-all-reviews-link-foot"]') || 
                       document.querySelector('a[href*="product-reviews"]') ||
                       document.querySelector('[data-hook="reviews-medley-footer"] a');
    
    if (reviewsLink) {
      return { 
        error: "No reviews found on current view. Please navigate to the reviews section first.",
        reviewsLink: reviewsLink.href 
      };
    }
  }

  // Extract review text and metadata
  reviewElements.forEach((element, index) => {
    const reviewText = element.innerText?.trim();
    if (reviewText && reviewText.length > 10) { // Filter out very short texts
      
      // Try to find associated rating for this review
      let rating = "";
      let reviewContainer = element.closest('[data-hook="review"]') || 
                           element.closest('.review') ||
                           element.closest('[data-hook="review-container"]');
      
      if (reviewContainer) {
        const ratingElement = reviewContainer.querySelector('[data-hook="review-star-rating"]') ||
                             reviewContainer.querySelector('.a-icon-alt') ||
                             reviewContainer.querySelector('[class*="star"]');
        if (ratingElement) {
          rating = ratingElement.getAttribute('title') || 
                   ratingElement.innerText || 
                   ratingElement.getAttribute('aria-label') || "";
        }
      }

      // Try to find reviewer name
      let reviewer = "";
      if (reviewContainer) {
        const reviewerElement = reviewContainer.querySelector('[data-hook="review-author"]') ||
                               reviewContainer.querySelector('.author') ||
                               reviewContainer.querySelector('[class*="author"]');
        if (reviewerElement) {
          reviewer = reviewerElement.innerText?.trim() || "";
        }
      }

      reviews.push({
        id: `review_${index + 1}`,
        text: reviewText,
        rating: rating,
        reviewer: reviewer
      });
    }
  });

  // If still no reviews, try alternative approach - look for any text that might be reviews
  if (reviews.length === 0) {
    const potentialReviews = document.querySelectorAll('span, p, div');
    const reviewKeywords = ['verified purchase', 'helpful', 'product', 'buy', 'recommend', 'quality', 'price', 'delivery'];
    
    potentialReviews.forEach((element, index) => {
      const text = element.innerText?.trim();
      if (text && text.length > 50 && text.length < 2000) {
        const lowerText = text.toLowerCase();
        const hasReviewKeywords = reviewKeywords.some(keyword => lowerText.includes(keyword));
        
        // Check if this looks like a review (has some review-like characteristics)
        if (hasReviewKeywords || lowerText.includes('star') || lowerText.includes('mrating')) {
          reviews.push({
            id: `potential_review_${index + 1}`,
            text: text,
            rating: "",
            reviewer: "Unknown"
          });
        }
      }
    });
  }

  return { 
    reviews: reviews.slice(0, 50), // Limit to first 50 reviews to avoid too much data
    totalFound: reviews.length,
    isAmazon: true 
  };
}



function scanTextForOffers(text, offers) {
  if (!text || text.length < 5) return;
  
  // Super aggressive pattern matching - exact matches from screenshot
  const exactPatterns = [
    // From screenshot: "Upto ₹3,379.17 EMI interest savings on select Credit Cards"
    {
      pattern: /Upto\s*₹([\d,]+\.?\d*)\s*EMI\s*interest\s*savings/gi,
      type: 'EMI Offer',
      priority: 2
    },
    // From screenshot: "Upto ₹587.00 discount on SBI Credit Cards, SBI D..."
    {
      pattern: /Upto\s*₹([\d,]+\.?\d*)\s*discount\s*on\s*SBI/gi,
      type: 'Bank Offer', 
      priority: 1
    },
    // From screenshot: "Upto ₹2,249.00 cashback as Amazon Pay Balance when..."
    {
      pattern: /Upto\s*₹([\d,]+\.?\d*)\s*cashback\s*as\s*Amazon\s*Pay/gi,
      type: 'Cashback',
      priority: 3
    },
    // General patterns
    {
      pattern: /₹([\d,]+\.?\d*)\s*EMI\s*interest\s*savings/gi,
      type: 'EMI Offer',
      priority: 2
    },
    {
      pattern: /₹([\d,]+\.?\d*)\s*discount.*?(SBI|HDFC|ICICI|Bank)/gi,
      type: 'Bank Offer',
      priority: 1
    },
    {
      pattern: /₹([\d,]+\.?\d*)\s*cashback/gi,
      type: 'Cashback',
      priority: 3
    },
    {
      pattern: /(\d+)%.*?(discount|off)/gi,
      type: 'Discount',
      priority: 4
    },
    {
      pattern: /save\s*up\s*to\s*(\d+)%/gi,
      type: 'Partner Offer',
      priority: 4
    }
  ];
  
  exactPatterns.forEach(({pattern, type, priority}) => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      if (match[1]) {
        const discount = match[0].includes('%') ? `${match[1]}%` : `₹${match[1]}`;
        const uniqueKey = `${type}-${discount}`;
        
        if (!offers.some(o => `${o.type}-${o.discount}` === uniqueKey)) {
          offers.push({
            type: type,
            title: `${type}: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: match[0].trim(),
            priority: priority
          });
          console.log(`Extracted offer: ${type} - ${discount} from text: ${match[0]}`);
        }
      }
    });
  });
  
  // Additional flexible patterns for edge cases
  const flexiblePatterns = [
    /No\s*Cost\s*EMI.*?₹([\d,]+\.?\d*)/gi,
    /EMI.*?₹([\d,]+\.?\d*).*?savings/gi,
    /Bank.*?₹([\d,]+\.?\d*).*?discount/gi,
    /₹([\d,]+\.?\d*).*?Amazon\s*Pay.*?Balance/gi
  ];
  
  flexiblePatterns.forEach(pattern => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      if (match[1]) {
        const discount = `₹${match[1]}`;
        let type = 'Special Offer';
        let priority = 4;
        
        const matchText = match[0].toLowerCase();
        if (matchText.includes('emi')) {
          type = 'EMI Offer';
          priority = 2;
        } else if (matchText.includes('bank')) {
          type = 'Bank Offer';
          priority = 1;
        } else if (matchText.includes('amazon pay') || matchText.includes('cashback')) {
          type = 'Cashback';
          priority = 3;
        }
        
        const uniqueKey = `${type}-${discount}`;
        if (!offers.some(o => `${o.type}-${o.discount}` === uniqueKey)) {
          offers.push({
            type: type,
            title: `${type}: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: match[0].trim(),
            priority: priority
          });
        }
      }
    });
  });
}

function scanOffersSection(offersSection, offers) {
  console.log('Scanning offers section...');
  
  // Look for individual offer cards/containers within the offers section
  const offerCards = offersSection.querySelectorAll('.a-box, .a-section, [data-csa-c-content-id], .offer-card, .offer-item, div[class*="offer"]');
  
  offerCards.forEach((card, index) => {
    const cardText = card.textContent || card.innerText || '';
    console.log(`Scanning offer card ${index}: ${cardText.substring(0, 100)}`);
    
    // Look for specific offer types based on headers or content
    if (cardText.toLowerCase().includes('no cost emi') || cardText.toLowerCase().includes('emi')) {
      extractEMIOffers(cardText, offers);
    }
    
    if (cardText.toLowerCase().includes('bank offer') || cardText.toLowerCase().includes('bank')) {
      extractBankOffers(cardText, offers);
    }
    
    if (cardText.toLowerCase().includes('cashback')) {
      extractCashbackOffers(cardText, offers);
    }
    
    if (cardText.toLowerCase().includes('partner offer') || cardText.toLowerCase().includes('gst')) {
      extractPartnerOffers(cardText, offers);
    }
    
    // General scan as fallback
    scanTextForOffers(cardText, offers);
  });
  
  // Also scan the entire offers section text
  const sectionText = offersSection.textContent || offersSection.innerText || '';
  scanTextForOffers(sectionText, offers);
}

function extractEMIOffers(text, offers) {
  const emiPatterns = [
    /upto\s*₹([\d,]+\.?\d*)\s*emi\s*interest\s*savings/gi,
    /no\s*cost\s*emi.*?₹([\d,]+\.?\d*)/gi,
    /₹([\d,]+\.?\d*)\s*emi\s*interest\s*savings/gi,
    /emi\s*starts\s*at\s*₹([\d,]+\.?\d*)/gi
  ];
  
  emiPatterns.forEach(pattern => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      if (match[1]) {
        const discount = `₹${match[1]}`;
        if (!offers.some(o => o.discount === discount && o.type.includes('EMI'))) {
          offers.push({
            type: 'No Cost EMI',
            title: `EMI Interest Savings: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: text.trim().substring(0, 150),
            priority: 2
          });
        }
      }
    });
  });
}

function extractBankOffers(text, offers) {
  const bankPatterns = [
    /upto\s*₹([\d,]+\.?\d*)\s*discount.*?(sbi|bank)/gi,
    /₹([\d,]+\.?\d*)\s*discount.*?(sbi|hdfc|icici|axis|kotak|bank)/gi,
    /(\d+)%.*?discount.*?(sbi|bank)/gi
  ];
  
  bankPatterns.forEach(pattern => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      const discountValue = match[1];
      if (discountValue) {
        const discount = match[0].includes('%') ? `${discountValue}%` : `₹${discountValue}`;
        if (!offers.some(o => o.discount === discount && o.type.includes('Bank'))) {
          offers.push({
            type: 'Bank Offer',
            title: `Bank Discount: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: text.trim().substring(0, 150),
            priority: 1
          });
        }
      }
    });
  });
}

function extractCashbackOffers(text, offers) {
  const cashbackPatterns = [
    /upto\s*₹([\d,]+\.?\d*)\s*cashback/gi,
    /₹([\d,]+\.?\d*)\s*cashback/gi,
    /(\d+)%.*?cashback/gi
  ];
  
  cashbackPatterns.forEach(pattern => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      const discountValue = match[1];
      if (discountValue) {
        const discount = match[0].includes('%') ? `${discountValue}%` : `₹${discountValue}`;
        if (!offers.some(o => o.discount === discount && o.type.includes('Cashback'))) {
          offers.push({
            type: 'Cashback',
            title: `Cashback: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: text.trim().substring(0, 150),
            priority: 3
          });
        }
      }
    });
  });
}

function extractPartnerOffers(text, offers) {
  const partnerPatterns = [
    /save\s*up\s*to\s*(\d+)%/gi,
    /(\d+)%\s*off.*?business/gi,
    /gst.*?invoice.*?(\d+)%/gi
  ];
  
  partnerPatterns.forEach(pattern => {
    const matches = [...text.matchAll(pattern)];
    matches.forEach(match => {
      if (match[1]) {
        const discount = `${match[1]}%`;
        if (!offers.some(o => o.discount === discount && o.type.includes('Partner'))) {
          offers.push({
            type: 'Partner Offer',
            title: `Business Discount: ${discount}`,
            discount: discount,
            numericValue: extractNumericValue(discount),
            description: text.trim().substring(0, 150),
            priority: 4
          });
        }
      }
    });
  });
}

function searchSpecificOfferPatterns(offers) {
  // Look for specific Amazon offer indicators
  const offerIndicators = [
    'lightning deal',
    'limited time deal', 
    'today\'s deal',
    'special offer',
    'bank offer',
    'emi offer',
    'cashback offer',
    'exchange offer'
  ];
  
  offerIndicators.forEach(indicator => {
    const elements = document.querySelectorAll(`*:contains("${indicator}")`);
    elements.forEach(element => {
      const text = element.textContent || element.innerText || '';
      if (text.length < 500) {
        scanTextForOffers(text, offers);
      }
    });
  });
}

function extractNumericValue(discountText) {
  if (!discountText) return 0;
  
  // Remove currency symbols and commas, extract numbers
  const cleanText = discountText.replace(/[₹,]/g, '');
  
  if (cleanText.includes('%')) {
    return parseFloat(cleanText.replace('%', '')) * 100; // Multiply by 100 to prioritize percentage discounts
  } else {
    return parseFloat(cleanText) || 0;
  }
}

function analyzeAndRankOffers(offers) {
  // Remove duplicates based on description
  const uniqueOffers = offers.filter((offer, index, self) =>
    index === self.findIndex(o => o.description === offer.description)
  );

  // Sort by numeric value (descending) and then by priority (ascending)
  const rankedOffers = uniqueOffers.sort((a, b) => {
    if (b.numericValue !== a.numericValue) {
      return b.numericValue - a.numericValue; // Higher discount first
    }
    return a.priority - b.priority; // Lower priority number = higher priority
  });

  return rankedOffers.slice(0, 10); // Return top 10 offers
}

chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
  if (req.type === "GET_ECOMMERCE_ITEMS") {
    const result = getAllEcommerceItems();
    sendResponse(result);
  }
  if (req.type === "GET_PAGE_HTML") {
    sendResponse({ html: document.documentElement.outerHTML });
  }
  if (req.type === "GET_AMAZON_REVIEWS") {
    const result = getAmazonReviews();
    sendResponse(result);
  }
});
