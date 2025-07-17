# 🎾 Tennis Prediction System API Documentation

## Overview

The Tennis Prediction System provides a comprehensive REST API for tennis match predictions, player analysis, and betting insights using advanced machine learning models.

**Base URL:** `http://localhost:5000` (development) or `https://your-domain.com` (production)

## Authentication

Currently, the API does not require authentication for public endpoints. Rate limiting is applied to API usage tracking endpoints.

## API Endpoints

### Health & Status

#### GET /api/health

Get system health status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T10:30:00Z",
  "uptime": "2h 15m",
  "components": {
    "ml_models": "available",
    "odds_api": "available", 
    "database": "available",
    "cache": "available"
  }
}
```

**Status Codes:**
- `200 OK` - System is operational
- `503 Service Unavailable` - System has critical issues

---

#### GET /api/stats

Get system statistics and performance metrics.

**Response:**
```json
{
  "predictions_today": 45,
  "accuracy_rate": 0.72,
  "api_calls_remaining": 450,
  "models_loaded": 5,
  "uptime_hours": 168.5
}
```

---

### Match Data

#### GET /api/matches

Get available tennis matches with odds and predictions.

**Query Parameters:**
- `surface` (optional): Filter by court surface (`Hard`, `Clay`, `Grass`)
- `tournament` (optional): Filter by tournament name
- `limit` (optional): Limit number of results (default: 20, max: 100)

**Example Request:**
```
GET /api/matches?surface=Clay&limit=10
```

**Response:**
```json
{
  "matches": [
    {
      "id": "match_001",
      "player1": "Novak Djokovic",
      "player2": "Rafael Nadal", 
      "tournament": "French Open",
      "surface": "Clay",
      "start_time": "2025-07-15T14:00:00Z",
      "odds": {
        "player1": 1.85,
        "player2": 1.95
      },
      "prediction": {
        "winner": "player1",
        "confidence": 0.75,
        "probability": 0.68
      }
    }
  ],
  "total": 1,
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_pages": 1
  }
}
```

---

### ML Predictions

#### POST /api/test-ml

Generate ML prediction for a tennis match.

**Request Body:**
```json
{
  "player_name": "Novak Djokovic",
  "opponent_name": "Rafael Nadal",
  "surface": "Clay",
  "tournament": "French Open",
  "player_rank": 1,
  "opponent_rank": 2,
  "player_age": 36,
  "opponent_age": 37
}
```

**Required Fields:**
- `player_name` (string): Name of the first player
- `opponent_name` (string): Name of the second player
- `surface` (string): Court surface (`Hard`, `Clay`, `Grass`)
- `player_rank` (integer): ATP/WTA ranking of first player
- `opponent_rank` (integer): ATP/WTA ranking of second player

**Optional Fields:**
- `tournament` (string): Tournament name
- `player_age` (integer): Age of first player
- `opponent_age` (integer): Age of second player

**Response:**
```json
{
  "prediction": 0.72,
  "confidence": "high",
  "winner": "Novak Djokovic",
  "individual_predictions": {
    "neural_network": 0.75,
    "xgboost": 0.70,
    "random_forest": 0.71,
    "gradient_boosting": 0.69,
    "logistic_regression": 0.73
  },
  "features_used": 23,
  "processing_time_ms": 145
}
```

**Status Codes:**
- `200 OK` - Prediction generated successfully
- `400 Bad Request` - Invalid input data
- `422 Unprocessable Entity` - Missing required fields
- `503 Service Unavailable` - ML models not available

---

#### POST /api/underdog-analysis

Analyze underdog potential for a tennis match.

**Request Body:**
```json
{
  "player_name": "Andrey Rublev",
  "opponent_name": "Novak Djokovic",
  "surface": "Hard",
  "tournament": "US Open",
  "player_rank": 8,
  "opponent_rank": 1
}
```

**Response:**
```json
{
  "is_underdog": true,
  "underdog_player": "Andrey Rublev",
  "value_rating": 0.85,
  "confidence": 0.72,
  "recommendation": "Strong underdog bet",
  "analysis": {
    "rank_advantage": false,
    "form_advantage": true,
    "surface_advantage": true,
    "h2h_advantage": false
  },
  "betting_suggestion": {
    "recommended": true,
    "max_stake_percentage": 5,
    "expected_value": 0.23
  }
}
```

---

### Value Betting

#### GET /api/value-bets

Get current value betting opportunities.

**Query Parameters:**
- `min_value` (optional): Minimum value rating (0.0-1.0)
- `surface` (optional): Filter by court surface
- `max_odds` (optional): Maximum acceptable odds

**Example Request:**
```
GET /api/value-bets?min_value=0.6&surface=Hard
```

**Response:**
```json
{
  "value_bets": [
    {
      "match_id": "match_005",
      "player": "Carlos Alcaraz",
      "opponent": "Daniil Medvedev",
      "tournament": "ATP Masters",
      "surface": "Hard",
      "odds": 2.10,
      "predicted_probability": 0.55,
      "value_rating": 0.76,
      "expected_value": 0.155,
      "confidence": "medium",
      "recommended_stake": 3.5
    }
  ],
  "total_opportunities": 1,
  "filters_applied": {
    "min_value": 0.6,
    "surface": "Hard"
  }
}
```

---

### Player Information

#### GET /api/player-info/{player_name}

Get detailed information about a specific player.

**Path Parameters:**
- `player_name`: URL-encoded player name

**Example Request:**
```
GET /api/player-info/Novak%20Djokovic
```

**Response:**
```json
{
  "player": {
    "name": "Novak Djokovic",
    "ranking": 1,
    "age": 36,
    "country": "Serbia",
    "playing_hand": "Right",
    "height": "188 cm",
    "weight": "77 kg"
  },
  "stats": {
    "career_titles": 98,
    "grand_slam_titles": 24,
    "win_rate_2024": 0.91,
    "prize_money": "$174,115,296"
  },
  "surface_performance": {
    "hard": 0.85,
    "clay": 0.79,
    "grass": 0.87
  },
  "recent_form": {
    "last_10_matches": 9,
    "current_streak": "W5",
    "form_rating": 0.92
  }
}
```

**Status Codes:**
- `200 OK` - Player information found
- `404 Not Found` - Player not found in database

---

### System Management

#### POST /api/manual-api-update

Manually trigger API data update (rate limited).

**Response:**
```json
{
  "status": "success",
  "message": "API update initiated",
  "timestamp": "2025-07-15T10:30:00Z",
  "usage": {
    "daily_requests": 3,
    "daily_limit": 8,
    "manual_requests": 1,
    "manual_limit": 5
  }
}
```

**Status Codes:**
- `200 OK` - Update initiated successfully
- `429 Too Many Requests` - Rate limit exceeded

---

#### GET /api/api-status

Get API usage and rate limiting status.

**Response:**
```json
{
  "api_usage": {
    "requests_used": 45,
    "requests_remaining": 455,
    "daily_usage": 3,
    "daily_limit": 8,
    "reset_time": "2025-07-16T00:00:00Z"
  },
  "status": "healthy",
  "next_scheduled_update": "2025-07-15T12:00:00Z",
  "last_update": "2025-07-15T08:00:00Z",
  "data_freshness": "2h 30m ago"
}
```

---

#### GET /api/api-economy-status

Get detailed API economy and usage statistics.

**Response:**
```json
{
  "monthly_budget": 500,
  "monthly_usage": 145,
  "daily_average": 4.8,
  "projected_monthly": 144,
  "efficiency_rating": 0.92,
  "cost_per_prediction": 0.003,
  "recommendations": [
    "Current usage is well within budget",
    "Consider increasing daily scheduled requests"
  ]
}
```

---

### Monitoring

#### GET /api/system-health

Get detailed system health and monitoring information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T10:30:00Z",
  "metrics": {
    "prediction_success_rate": 0.94,
    "api_response_time_ms": 850,
    "memory_usage_mb": 512,
    "active_models": 5,
    "error_count_last_hour": 0
  },
  "issues": [],
  "recent_alerts": 0,
  "uptime": "168h 15m"
}
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Required field 'player_rank' is missing",
    "details": {
      "field": "player_rank",
      "expected_type": "integer",
      "received": null
    },
    "timestamp": "2025-07-15T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

- `INVALID_INPUT` - Invalid request data
- `MISSING_FIELD` - Required field missing
- `SERVICE_UNAVAILABLE` - External service not available
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INTERNAL_ERROR` - Unexpected server error
- `MODEL_ERROR` - ML model prediction failed

---

## Rate Limiting

The API implements several rate limiting mechanisms:

### Global Rate Limits
- **General endpoints**: 100 requests per minute
- **ML prediction endpoints**: 20 requests per minute
- **Manual update endpoints**: 5 requests per day

### Headers
All responses include rate limiting headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642678800
X-RateLimit-Window: 60
```

---

## WebSocket Support

Real-time updates are available via WebSocket connections:

### Connection
```javascript
const ws = new WebSocket('ws://localhost:5000/ws');
```

### Event Types
- `match_update` - Live match data updates
- `prediction_complete` - New ML prediction available
- `odds_change` - Betting odds updated
- `system_alert` - System status alerts

### Example Message
```json
{
  "type": "prediction_complete",
  "data": {
    "match_id": "match_001",
    "prediction": 0.75,
    "confidence": "high",
    "timestamp": "2025-07-15T10:30:00Z"
  }
}
```

---

## SDK Examples

### Python
```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"
headers = {"Content-Type": "application/json"}

# Make a prediction
match_data = {
    "player_name": "Novak Djokovic",
    "opponent_name": "Rafael Nadal",
    "surface": "Clay",
    "player_rank": 1,
    "opponent_rank": 2
}

response = requests.post(
    f"{BASE_URL}/api/test-ml",
    headers=headers,
    data=json.dumps(match_data)
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Prediction: {prediction['prediction']:.2f}")
    print(f"Confidence: {prediction['confidence']}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript
```javascript
// Async function to get match prediction
async function getPrediction(matchData) {
    try {
        const response = await fetch('http://localhost:5000/api/test-ml', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(matchData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const prediction = await response.json();
        return prediction;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// Usage
const matchData = {
    player_name: "Carlos Alcaraz",
    opponent_name: "Jannik Sinner", 
    surface: "Hard",
    player_rank: 2,
    opponent_rank: 4
};

getPrediction(matchData)
    .then(prediction => {
        console.log('Prediction:', prediction.prediction);
        console.log('Confidence:', prediction.confidence);
    })
    .catch(error => {
        console.error('Failed to get prediction:', error);
    });
```

### cURL
```bash
# Health check
curl -X GET http://localhost:5000/api/health

# Make prediction
curl -X POST http://localhost:5000/api/test-ml \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Stefanos Tsitsipas",
    "opponent_name": "Alexander Zverev",
    "surface": "Clay",
    "player_rank": 5,
    "opponent_rank": 6
  }'

# Get value bets
curl -X GET "http://localhost:5000/api/value-bets?min_value=0.7&surface=Hard"
```

---

## Data Models

### Match Data Model
```json
{
  "player_name": "string",
  "opponent_name": "string", 
  "surface": "Hard|Clay|Grass",
  "tournament": "string",
  "player_rank": "integer (1-1000)",
  "opponent_rank": "integer (1-1000)",
  "player_age": "integer (optional)",
  "opponent_age": "integer (optional)"
}
```

### Prediction Response Model
```json
{
  "prediction": "float (0.0-1.0)",
  "confidence": "low|medium|high",
  "winner": "string",
  "individual_predictions": {
    "neural_network": "float",
    "xgboost": "float", 
    "random_forest": "float",
    "gradient_boosting": "float",
    "logistic_regression": "float"
  },
  "features_used": "integer",
  "processing_time_ms": "integer"
}
```

---

## Best Practices

### Request Guidelines
1. **Always include required fields** - Check API documentation for required parameters
2. **Use appropriate HTTP methods** - GET for data retrieval, POST for predictions
3. **Handle rate limits gracefully** - Implement exponential backoff for rate limit responses
4. **Validate input data** - Ensure rankings are 1-1000, surfaces are valid options
5. **Check response status codes** - Don't assume all requests succeed

### Performance Optimization
1. **Cache results when appropriate** - ML predictions for same inputs don't change frequently
2. **Batch requests when possible** - Use bulk endpoints for multiple predictions
3. **Monitor your usage** - Stay within rate limits and API quotas
4. **Use WebSocket for real-time data** - More efficient than polling for live updates

### Error Handling
1. **Implement retry logic** - For temporary failures (5xx errors)
2. **Handle validation errors** - Parse error messages for specific field issues
3. **Monitor API status** - Check `/api/health` before making critical requests
4. **Log errors appropriately** - Include request ID for debugging

---

## Support & Contact

For API support, issues, or feature requests:

- **Documentation**: This README and inline API documentation
- **Health Status**: Monitor `/api/health` endpoint
- **System Logs**: Check application logs for detailed error information

## Changelog

### Version 1.0 (Current)
- Initial API release
- ML prediction endpoints
- Value betting analysis
- Real-time monitoring
- Rate limiting and usage tracking