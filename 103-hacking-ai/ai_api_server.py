#!/usr/bin/env python3
"""
Secure AI API Server - Sentiment Analysis Service
Demonstrates: Authentication, Rate Limiting, Input Validation, Logging
"""

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
import time
import hashlib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key-change-in-production'

# Rate limiter configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Simple API key storage (use database in production)
VALID_API_KEYS = {
    'demo-key-12345': {'user': 'alice', 'tier': 'premium'},
    'demo-key-67890': {'user': 'bob', 'tier': 'basic'},
    'test-key-99999': {'user': 'testuser', 'tier': 'basic'}
}

# Tier-based rate limits
TIER_LIMITS = {
    'premium': "200 per hour",
    'basic': "50 per hour"
}

# Security patterns
SUSPICIOUS_PATTERNS = [
    r'ignore\s+(previous|above|prior)\s+instructions',
    r'system\s*:',
    r'<\s*script',
    r'eval\s*\(',
    r'exec\s*\(',
    r'__import__',
    r'DROP\s+TABLE',
    r'DELETE\s+FROM',
]

# PII patterns (simple examples)
PII_PATTERNS = [
    r'\d{3}-\d{2}-\d{4}',  # SSN
    r'\d{16}',  # Credit card (simplified)
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email
]

def authenticate_request():
    """Validate API key from request header."""
    api_key = request.headers.get('X-API-Key')

    if not api_key:
        return None, {'error': 'Missing API key'}, 401

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        return None, {'error': 'Invalid API key'}, 403

    user_info = VALID_API_KEYS[api_key]
    return user_info, None, None

def sanitize_input(text, max_length=1000):
    """Sanitize and validate input text."""
    if not text or not isinstance(text, str):
        return None, "Input must be a non-empty string"

    # Length check
    if len(text) > max_length:
        return None, f"Input exceeds maximum length of {max_length} characters"

    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())

    # Check for suspicious patterns
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected: {pattern}")
            return None, f"Input contains suspicious pattern: {pattern}"

    # Check for PII
    pii_found = []
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            pii_found.append(pattern)

    if pii_found:
        logger.warning(f"PII detected in input: {pii_found}")
        return None, "Input contains potential PII (email, SSN, etc.)"

    return text, None

def simple_sentiment_analysis(text):
    """
    Simple rule-based sentiment analysis.
    In production, this would call a real ML model.
    """
    # Positive/negative word lists
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                     'fantastic', 'love', 'perfect', 'best', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 
                     'hate', 'poor', 'disappointing', 'useless', 'fail']

    text_lower = text.lower()

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count

    if total == 0:
        sentiment = 'neutral'
        confidence = 0.5
    elif pos_count > neg_count:
        sentiment = 'positive'
        confidence = min(0.9, 0.5 + (pos_count / (total + 1)))
    else:
        sentiment = 'negative'
        confidence = min(0.9, 0.5 + (neg_count / (total + 1)))

    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 3),
        'positive_signals': pos_count,
        'negative_signals': neg_count
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ai-sentiment-api',
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
@limiter.limit(lambda: TIER_LIMITS.get(authenticate_request()[0].get('tier', 'basic'), "50 per hour"))
def analyze_sentiment():
    """
    Sentiment analysis endpoint with security controls.

    Request: {"text": "Your input text here"}
    Response: {"sentiment": "positive/negative/neutral", "confidence": 0.8}
    """
    start_time = time.time()

    # Authentication
    user_info, error_response, status_code = authenticate_request()
    if error_response:
        return jsonify(error_response), status_code

    # Parse request
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        input_text = data['text']
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        return jsonify({'error': 'Invalid JSON'}), 400

    # Input sanitization
    sanitized_text, error = sanitize_input(input_text, max_length=1000)
    if error:
        logger.warning(f"Input validation failed for user {user_info['user']}: {error}")
        return jsonify({'error': error}), 400

    # Perform sentiment analysis
    try:
        result = simple_sentiment_analysis(sanitized_text)

        # Add metadata
        result['user'] = user_info['user']
        result['tier'] = user_info['tier']
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)

        # Log successful request
        logger.info(f"Analysis complete - User: {user_info['user']}, "
                   f"Sentiment: {result['sentiment']}, "
                   f"Time: {result['processing_time_ms']}ms")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API statistics (demo endpoint)."""
    user_info, error_response, status_code = authenticate_request()
    if error_response:
        return jsonify(error_response), status_code

    # In production, this would query a database
    return jsonify({
        'total_requests': 1337,
        'avg_processing_time_ms': 42.5,
        'error_rate': 0.02,
        'uptime_hours': 168
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit exceeded: {get_remote_address()}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

if __name__ == '__main__':
    logger.info("Starting AI API Server on http://127.0.0.1:5000")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /api/analyze - Sentiment analysis")
    logger.info("  GET  /api/stats - API statistics")
    app.run(debug=True, host='127.0.0.1', port=5000)
