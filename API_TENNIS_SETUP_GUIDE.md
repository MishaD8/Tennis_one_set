# API-Tennis.com Integration Setup Guide

## Overview

Your tennis backend system now includes comprehensive integration with API-Tennis.com, providing access to real-time tennis data including:

- ✅ Professional tournament schedules (ATP/WTA only)
- ✅ Live match data and scores
- ✅ Player rankings and statistics
- ✅ Betting odds from multiple bookmakers
- ✅ Head-to-head match histories
- ✅ Tournament standings

## Setup Instructions

### 1. Get Your API Key

1. Visit https://api-tennis.com/
2. Create an account
3. Purchase a subscription plan
4. Copy your API key from the dashboard

### 2. Configure Environment Variable

Add your API key to your `.env` file:

```bash
# API-Tennis.com Configuration (Primary tennis data source)
API_TENNIS_KEY=your_actual_api_key_here
```

### 3. Restart Your Backend

After adding the API key, restart your tennis backend:

```bash
python3 tennis_backend.py
```

## Available Endpoints

Once configured, you'll have access to these new API endpoints:

### Integration Status
```
GET /api/api-tennis/status
```
Returns current integration status and configuration

### Connection Test
```
GET /api/api-tennis/test-connection
```
Tests API connectivity and key validity

### Tournaments
```
GET /api/api-tennis/tournaments
```
Retrieves professional ATP/WTA tournaments

### Matches
```
GET /api/api-tennis/matches?include_live=true&days_ahead=2
```
Retrieves current and upcoming matches

### Player-Specific Matches
```
GET /api/api-tennis/player/Novak Djokovic/matches?days_ahead=30
```
Retrieves matches for a specific player

### Match Odds
```
GET /api/api-tennis/match/123456/odds
```
Retrieves betting odds for a specific match

### Enhanced Data Collection
```
GET /api/api-tennis/enhanced?days_ahead=2
```
Comprehensive data from multiple sources combined

### Cache Management
```
POST /api/api-tennis/clear-cache
```
Clears local cache (requires API key authentication)

## Integration Features

### Professional Tournament Filtering
The integration automatically filters to show only professional ATP/WTA events, excluding:
- Junior tournaments
- College/university events
- Amateur competitions
- UTR and PTT events

### Rate Limiting
- Conservative 50 requests per minute limit
- Built-in request throttling
- Automatic retry logic

### Local Caching
- 15-minute cache duration
- Reduces API calls
- Improves response times
- Configurable cache directory

### Data Normalization
All API-Tennis data is normalized to match your existing Universal Collector format for seamless integration with your ML models.

### Error Handling
- Comprehensive exception handling
- Graceful fallback to other data sources
- Detailed error logging
- Connection resilience

## Testing the Integration

### Basic Test Script
Run the included test script:

```bash
python3 test_api_tennis_integration.py
```

### Test Direct Client
```bash
python3 test_api_tennis_integration.py direct
```

### Test Both Client and Server
```bash
python3 test_api_tennis_integration.py both
```

## Verification Steps

1. **Check Status Endpoint:**
   ```bash
   curl http://localhost:5001/api/api-tennis/status
   ```

2. **Test Connection:**
   ```bash
   curl http://localhost:5001/api/api-tennis/test-connection
   ```

3. **Get Tournaments:**
   ```bash
   curl http://localhost:5001/api/api-tennis/tournaments
   ```

## Expected Behavior

### With Valid API Key
- ✅ Real tournament data
- ✅ Live match updates
- ✅ Current player rankings
- ✅ Betting odds integration
- ✅ Professional event filtering

### Without API Key
- ⚠️  Integration marked as unavailable
- 🔄 Fallback to existing data sources
- 📊 System continues to work with simulated data
- 🛡️  No API calls attempted

## Benefits for Your Tennis Prediction System

### Enhanced Data Quality
- Real tournament schedules instead of simulated data
- Accurate player rankings and seedings
- Live score updates for in-play analysis

### Better Underdog Detection
- Access to real odds from multiple bookmakers
- Professional tournament filtering focuses on ATP/WTA
- Improved ranking accuracy for players ranked 50-300

### ML Model Improvements
- Higher quality training data
- Real match outcomes for model validation
- Professional tournament context for predictions

## Troubleshooting

### Common Issues

1. **"API-Tennis not configured"**
   - Ensure API_TENNIS_KEY is set in your .env file
   - Restart the backend after adding the key

2. **"Connection failed"**
   - Verify your API key is correct
   - Check internet connectivity
   - Confirm API quota hasn't been exceeded

3. **Empty results**
   - Professional filtering may exclude some events
   - Check date range parameters
   - Verify tournament schedule

### Debug Information

Check the logs for detailed error messages:
- API request/response details
- Rate limiting status
- Cache hit/miss information
- Professional tournament filtering results

## Integration Architecture

```
Your Tennis Backend
├── api_tennis_integration.py      (Core API client)
├── api_tennis_data_collector.py   (Data normalization)
├── routes.py                      (API endpoints)
├── config.py                     (Configuration)
└── test_api_tennis_integration.py (Testing)
```

The integration follows your existing architecture patterns and integrates seamlessly with your Universal Collector system.

## Next Steps

1. **Configure your API key** in the .env file
2. **Test the integration** using the provided scripts
3. **Monitor the dashboard** for real tennis data
4. **Verify ML predictions** improve with real data
5. **Check rate limiting** and adjust if needed

## Support

- Check the API documentation: `/docs/API_DOCUMENTATION.md`
- Review system logs for detailed error information
- Test individual endpoints using the provided test script
- Verify configuration using the status endpoint

Your API-Tennis.com integration is now ready to provide real-time professional tennis data to enhance your underdog detection and ML prediction system!