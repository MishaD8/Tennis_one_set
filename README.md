# Tennis One Set - Tennis Prediction System

> A comprehensive machine learning system for tennis underdog prediction and match analysis.

## Overview

Tennis One Set is an advanced tennis prediction system that specializes in identifying strong underdogs likely to win the second set in ATP and WTA singles tournaments. The system focuses on players ranked 10-300 and uses machine learning models to improve prediction accuracy.


AREAS FOR IMPROVEMENT FOR THE SYSTEM:


- First set momentum indicators (momentum score, break points saved/converted)
- Player fatigue modeling (previous match dates, travel distance)
- Weather/court conditions (if available)
- In-match form (service percentage, unforced errors in first set)
- Psychological pressure indicators (ranking pressure, tournament importance)# Implement these improvements

1. Neural Networks with LSTM for sequential match data
2. Gradient Boosting with custom tennis loss functions
3. Ensemble methods with dynamic weighting based on match context
4. Real-time model updating during matches

# Enhanced data collection

- Integrate tennis-specific APIs (TennisBot, Ultimate Tennis Statistics)
- Add betting market data for market efficiency analysis
- Include player interview/social media sentiment analysis
- Weather API integration for outdoor tournaments

# Live prediction pipeline

1. WebSocket connection to live match feeds
2. Real-time feature calculation during first set
3. Prediction updates every game/point
4. Dynamic confidence scoring based on match state

# Target APIs for live data
- WTA/ATP live scoring APIs
- Tennis-specific statistics providers
- Weather APIs for outdoor matches
- Betting exchange APIs for market data# Neural network for sequential data
import tensorflow as tf

class TennisSequentialModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        # Process sequential match data
        lstm_out = self.lstm(inputs)
        attended = self.attention([lstm_out, lstm_out])
        return self.dense(attended)# Add these high-value features
def create_advanced_features(match_data):
    features = {}
    
    # First set momentum features
    features['first_set_momentum'] = calculate_momentum_score(match_data)
    features['break_point_efficiency'] = match_data['bp_converted'] / match_data['bp_faced']
    features['service_hold_percentage'] = match_data['service_games_won'] / match_data['service_games']
    
    # Fatigue and form features
    features['days_since_last_match'] = (current_date - last_match_date).days
    features['recent_form_trend'] = calculate_form_trend(recent_matches)
    features['surface_adaptation'] = matches_on_surface_last_30_days
    
    # Psychological features
    features['ranking_pressure'] = abs(target_ranking - current_ranking)
    features['tournament_importance'] = tournament_category_weight
    features['upset_potential'] = ranking_gap * form_factor
    
    return features# WebSocket integration for live data
import websockets
import asyncio

class LiveMatchTracker:
    def __init__(self):
        self.active_matches = {}
        self.prediction_engine = YourPredictionEngine()
    
    async def track_live_matches(self):
        uri = "wss://api.api-tennis.com/tennis/live"
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                match_update = json.loads(data)
                
                if self.is_first_set_complete(match_update):
                    prediction = self.make_second_set_prediction(match_update)
                    await self.send_notification(prediction)# Ensemble with dynamic weighting
class DynamicTennisEnsemble:
    def __init__(self, models):
        self.models = models
        self.weight_calculator = ContextualWeightCalculator()
    
    def predict(self, features, match_context):
        # Calculate dynamic weights based on match context
        weights = self.weight_calculator.get_weights(match_context)
        
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict_proba(features)
            predictions.append(pred * weights[model_name])
        
        return np.average(predictions, axis=0)
        
# Track these metrics

1. Prediction accuracy by tournament tier
2. ROI by surface type and ranking gap
3. Model calibration (predicted vs actual probabilities)
4. Market efficiency (edge over bookmaker odds)
5. Live vs historical performance consistencypredictions

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- Redis 6+
- Required API keys (see [Configuration](#configuration))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Tennis_one_set
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp config/config.json.example config/config.json
   # Edit config.json with your API keys and settings
   ```

4. **Initialize database**
   ```bash
   python src/data/database_setup.py
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The application will be available at `http://localhost:5000`

## System Architecture

```
Tennis_one_set/
├── src/                    # Source code
│   ├── api/               # Flask API application
│   ├── data/              # Data collection and processing
│   ├── models/            # Machine learning models
│   ├── utils/             # Utility functions
│   └── tests/             # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
├── tennis_models/         # Trained ML models
└── main.py               # Application entry point
```

## Key Features

- **Second Set Focus**: Specialized ML models for second set # Add these tennis-specific features
- **Professional Data**: ATP/WTA singles tournaments only
- **Ranking Analysis**: Focus on players ranked 50-300
- **Multi-Source Data**: Integration with multiple tennis APIs
- **Real-time Predictions**: Live match analysis and underdog detection
- **Comprehensive Testing**: Full test suite with organized structure

## Documentation

### Quick Reference
- [API Documentation](docs/API_DOCUMENTATION.md) - Complete API endpoint reference
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Test Documentation](src/tests/README.md) - Test suite organization

### Implementation Details
- [System Analysis](docs/SYSTEM_ANALYSIS_REPORT.md) - Comprehensive system overview
- [ML Implementation](docs/IMPLEMENTATION_SUMMARY.md) - Machine learning implementation details
- [Security Guide](docs/SECURITY_DEPLOYMENT_GUIDE.md) - Security configuration

### Setup Guides
- [API-Tennis Setup](docs/API_TENNIS_SETUP_GUIDE.md) - API-Tennis.com integration
- [Frontend Structure](docs/FRONTEND_STRUCTURE.md) - Frontend/backend separation

### Project Status
- [Cleanup Summary](PROJECT_CLEANUP_SUMMARY.md) - Recent cleanup and consolidation
- [Restructure Summary](RESTRUCTURE_SUMMARY.md) - Project restructuring details

## Configuration

### Required API Keys

1. **API-Tennis.com**: Primary tennis data source
   ```json
   {
     "API_TENNIS_KEY": "your_api_tennis_key"
   }
   ```

2. **The Odds API**: Betting odds data (500 requests/month)
   ```json
   {
     "ODDS_API_KEY": "your_odds_api_key"
   }
   ```

3. **RapidAPI Tennis**: Rankings and statistics (50 requests/day)
   ```json
   {
     "RAPIDAPI_KEY": "your_rapidapi_key"
   }
   ```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/tennis_db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Application
FLASK_ENV=production
DEBUG=false
```

## Usage Examples

### Basic Prediction
```python
from src.models.comprehensive_tennis_prediction_service import ComprehensiveTennisPredictionService

service = ComprehensiveTennisPredictionService()
result = service.predict_second_set_underdog(
    player1_name="Player A (Rank 180)",
    player2_name="Player B (Rank 120)"
)
print(f"Underdog probability: {result['underdog_probability']}")
```

### API Endpoints
```bash
# System health check
curl http://localhost:5000/api/health

# Get current matches
curl http://localhost:5000/api/matches

# Underdog analysis
curl http://localhost:5000/api/underdog-analysis
```

## Testing

The project includes a comprehensive test suite organized into categories:

```bash
# Run all tests
pytest src/tests/

# Run specific test categories
pytest src/tests/api/          # API tests
pytest src/tests/integration/  # Integration tests
pytest src/tests/unit/         # Unit tests
```

See [Test Documentation](src/tests/README.md) for detailed testing information.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add new feature'`)
7. Push to the branch (`git push origin feature/new-feature`)
8. Create a Pull Request

## License

This project is proprietary software. All rights reserved.

## Support

For technical support or questions:

1. Check the [documentation](docs/) first
2. Review [system status reports](docs/SYSTEM_ANALYSIS_REPORT.md)
3. Check the [troubleshooting guide](docs/DEPLOYMENT_GUIDE.md#troubleshooting)
4. Contact the development team

---

**Last Updated**: August 2025  
**Version**: 1.0  
**Status**: Production Ready ✅