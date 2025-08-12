# WebSocket Real-Time Tennis Betting System

## Overview

This system provides a complete real-time tennis betting infrastructure that integrates WebSocket live data streaming with ML predictions and automated betting. The system is designed for production use with robust error handling, data validation, and risk management.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API-Tennis    │───▶│  WebSocket       │───▶│  Data Quality   │
│   WebSocket     │    │  Client          │    │  Validator      │
│   Feed          │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Automated     │◀───│  Real-Time ML    │◀───│  Live Data      │
│   Betting       │    │  Pipeline        │    │  Buffer         │
│   Engine        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Betfair       │    │  Prediction      │    │  Feature        │
│   API           │    │  Engine          │    │  Engineering    │
│   Integration   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. WebSocket Client (`websocket_tennis_client.py`)

**Purpose**: Maintains real-time connection to API-Tennis WebSocket feed for live match data.

**Key Features**:
- Auto-reconnection with exponential backoff
- Connection state management
- Event filtering and routing
- SSL/TLS secure connections
- Rate limiting and connection pooling

**Usage**:
```python
from websocket_tennis_client import TennisWebSocketClient

client = TennisWebSocketClient(api_key="your_api_key")
client.add_event_callback(handle_live_event)
client.start_threaded()
```

### 2. Real-Time ML Pipeline (`realtime_ml_pipeline.py`)

**Purpose**: Processes live data through ML models to generate predictions for betting decisions.

**Key Features**:
- Live data buffering and feature engineering
- Multiple ML model integration
- Prediction confidence scoring
- Real-time performance monitoring

**Key Classes**:
- `LiveDataBuffer`: Stores and processes recent match events
- `MLPredictionProcessor`: Handles ML model execution
- `RealTimeMLPipeline`: Main coordinator

### 3. Data Quality Validator (`realtime_data_validator.py`)

**Purpose**: Ensures data integrity and quality for reliable betting decisions.

**Key Features**:
- Field-level validation
- Cross-field consistency checks
- Historical data validation
- Business logic verification
- Quality scoring and alerting

**Validation Metrics**:
- Completeness (all required fields present)
- Accuracy (data values are correct)
- Consistency (data doesn't contradict itself)
- Timeliness (data is recent and relevant)
- Validity (data follows expected formats)

### 4. Prediction Engine (`realtime_prediction_engine.py`)

**Purpose**: Detects when to trigger predictions and manages prediction workflow.

**Key Features**:
- Multiple trigger types (score changes, break points, etc.)
- Momentum tracking and analysis
- Priority-based prediction queue
- Configurable trigger conditions

**Trigger Types**:
- `MATCH_START`: Beginning of match
- `SCORE_CHANGE`: Set or game score changes
- `BREAK_POINT`: Break point situations
- `SET_POINT`: Set point situations
- `MOMENTUM_SHIFT`: Significant momentum changes
- `PERIODIC`: Regular interval updates

### 5. Automated Betting Engine (`automated_betting_engine.py`)

**Purpose**: Executes betting decisions based on ML predictions with comprehensive risk management.

**Key Features**:
- Betfair API integration
- Kelly Criterion stake calculation
- Multi-level risk management
- Position tracking and settlement
- P&L monitoring

**Risk Management**:
- Maximum stake per bet/match/day/week limits
- Confidence and edge thresholds
- Stop-loss and profit-taking rules
- Bankroll protection mechanisms

## Setup and Configuration

### 1. Environment Variables

Create a `.env` file with the following variables:

```bash
# API-Tennis WebSocket Configuration
API_TENNIS_KEY=your_api_tennis_key

# Betfair API Configuration (for live betting)
BETFAIR_APP_KEY=your_betfair_app_key
BETFAIR_USERNAME=your_betfair_username
BETFAIR_PASSWORD=your_betfair_password

# System Configuration
FLASK_ENV=production
REDIS_URL=redis://localhost:6379
```

### 2. Dependencies

Install required packages:

```bash
pip install websockets asyncio numpy pandas scikit-learn
pip install redis celery flask requests
```

### 3. Configuration

The system uses configuration classes in `config.py`. Key settings:

```python
# Risk Management
DEFAULT_STAKE = 10.0
MAX_STAKE = 100.0
PREDICTION_CONFIDENCE_THRESHOLD = 0.6

# API Limits
DAILY_API_LIMIT = 8
MONTHLY_API_LIMIT = 500
```

## Usage Examples

### Basic WebSocket Connection

```python
from websocket_tennis_client import TennisWebSocketClient

# Create client
client = TennisWebSocketClient(api_key="your_key")

# Add event handler
def handle_event(event):
    print(f"Match: {event.first_player} vs {event.second_player}")
    print(f"Score: {event.final_result}")

client.add_event_callback(handle_event)

# Start connection
client.start_threaded()
```

### Complete System Integration

```python
from realtime_prediction_engine import PredictionEngine, PredictionConfig
from automated_betting_engine import AutomatedBettingEngine, RiskManagementConfig

# Configure prediction engine
pred_config = PredictionConfig(
    enabled_triggers=[PredictionTrigger.MATCH_START, PredictionTrigger.SCORE_CHANGE],
    min_confidence_threshold=0.7,
    max_predictions_per_match=10
)

# Configure betting engine
risk_config = RiskManagementConfig.conservative()

# Create engines
prediction_engine = PredictionEngine(pred_config)
betting_engine = AutomatedBettingEngine(risk_config, initial_bankroll=1000.0)

# Initialize and connect
betting_engine.initialize(prediction_engine)

# Start systems
prediction_engine.start(api_key="your_key")
betting_engine.start()
```

### Data Quality Monitoring

```python
from realtime_data_validator import DataQualityMonitor

monitor = DataQualityMonitor(alert_threshold=0.7)

# Add alert handler
def handle_quality_alert(alert_type, data):
    print(f"QUALITY ALERT: {alert_type}")
    # Send notification, log to file, etc.

monitor.add_alert_callback(handle_quality_alert)

# Process events
report = monitor.process_event(live_event)
print(f"Quality Score: {report.overall_score}")
```

## Testing and Validation

### Running Tests

Execute the comprehensive test suite:

```bash
python test_websocket_integration.py
```

Test coverage includes:
- WebSocket connection and message handling
- Data validation and quality checks
- ML pipeline processing
- Prediction trigger detection
- Risk management rules
- End-to-end workflow integration

### System Demonstration

Run the complete system demonstration:

```bash
python websocket_system_demo.py
```

This provides:
- Live or simulated data processing
- Real-time statistics monitoring
- Performance metrics
- Complete workflow demonstration

## Production Deployment

### 1. Security Considerations

- Use environment variables for API keys
- Implement proper SSL/TLS for all connections
- Set up proper firewall rules
- Use encrypted configuration storage

### 2. Monitoring and Alerting

- Set up logging aggregation (ELK stack)
- Configure metrics monitoring (Prometheus/Grafana)
- Implement health checks
- Set up alert notifications

### 3. Scalability

- Use Redis for distributed caching
- Implement horizontal scaling with load balancers
- Set up database replication
- Use message queues for high-volume processing

### 4. Backup and Recovery

- Automated database backups
- Configuration backup
- Model versioning and rollback
- Disaster recovery procedures

## Performance Optimization

### 1. WebSocket Optimization

- Connection pooling
- Message batching
- Compression for large payloads
- Optimal reconnection strategies

### 2. ML Pipeline Optimization

- Model caching and preloading
- Feature computation optimization
- Parallel prediction processing
- Result caching strategies

### 3. Database Optimization

- Proper indexing strategies
- Connection pooling
- Query optimization
- Data archiving policies

## Risk Management

### 1. Technical Risks

- **Data Quality Issues**: Comprehensive validation and quality monitoring
- **API Failures**: Retry mechanisms and fallback strategies
- **Model Performance**: Continuous monitoring and retraining
- **System Downtime**: Redundancy and failover mechanisms

### 2. Financial Risks

- **Stake Limits**: Multiple levels of stake controls
- **Loss Limits**: Daily, weekly, and monthly loss limits
- **Position Management**: Maximum exposure controls
- **Model Validation**: Backtesting and performance tracking

### 3. Operational Risks

- **Human Error**: Automated processes with minimal manual intervention
- **Configuration Errors**: Validation and testing of all configuration changes
- **Security Breaches**: Comprehensive security measures and monitoring
- **Regulatory Compliance**: Adherence to gambling regulations

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check API key validity
   - Verify network connectivity
   - Review SSL certificate issues
   - Check rate limiting

2. **Poor Data Quality**
   - Review validation rules
   - Check data source reliability
   - Analyze error patterns
   - Adjust quality thresholds

3. **ML Prediction Issues**
   - Verify model loading
   - Check feature engineering
   - Review input data format
   - Monitor prediction confidence

4. **Betting Execution Problems**
   - Verify Betfair API credentials
   - Check account balance
   - Review risk management rules
   - Validate market availability

### Logging and Debugging

The system provides comprehensive logging at multiple levels:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_system.log'),
        logging.StreamHandler()
    ]
)
```

Key log files:
- `websocket_client.log`: WebSocket connection and message logs
- `ml_pipeline.log`: ML processing and prediction logs
- `betting_engine.log`: Betting decision and execution logs
- `data_quality.log`: Data validation and quality alerts

## API Documentation

### WebSocket Message Format

Incoming messages from API-Tennis follow this format:

```json
{
  "event_key": 11997372,
  "event_date": "2024-11-07",
  "event_time": "09:10",
  "event_first_player": "P. Verbin",
  "first_player_key": 13391,
  "event_second_player": "M. Kamrowski",
  "second_player_key": 15215,
  "event_final_result": "0 - 0",
  "event_game_result": "0 - 0",
  "event_serve": "Second Player",
  "event_winner": null,
  "event_status": "Set 1",
  "event_type_type": "Itf Men Singles",
  "tournament_name": "ITF M15 Sharm ElSheikh 15 Men",
  "tournament_key": 8153,
  "tournament_round": null,
  "tournament_season": "2024",
  "event_live": "1",
  "pointbypoint": [...],
  "scores": [...],
  "statistics": [...]
}
```

### ML Prediction Format

ML predictions follow this standardized format:

```json
{
  "match_id": 12345,
  "prediction": {
    "winner": "Player 1",
    "player_1_win_probability": 0.65,
    "player_2_win_probability": 0.35,
    "confidence": 0.78,
    "recommended_stake": 25.0
  },
  "model_used": "enhanced_ml_orchestrator",
  "processing_time": 0.15,
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Betting Opportunity Format

Betting opportunities are structured as:

```json
{
  "match_id": 12345,
  "bet_type": "match_winner",
  "selection": "Player 1",
  "predicted_odds": 1.54,
  "market_odds": 1.85,
  "confidence": 0.78,
  "edge": 0.15,
  "recommended_stake": 25.0,
  "reasoning": "ML confidence 0.78, edge 0.15"
}
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Daily**
   - Monitor system health
   - Review betting performance
   - Check data quality metrics
   - Validate API connectivity

2. **Weekly**
   - Analyze prediction accuracy
   - Review risk management metrics
   - Update model performance
   - Check system logs

3. **Monthly**
   - Model retraining and validation
   - Configuration review
   - Performance optimization
   - Security audit

### Getting Support

For issues or questions:

1. Check this documentation
2. Review system logs
3. Run diagnostic tests
4. Contact system administrator

This WebSocket integration provides a robust foundation for real-time tennis betting with comprehensive monitoring, validation, and risk management capabilities.