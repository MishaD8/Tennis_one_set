# Tennis One Set - Tennis Prediction System

> A comprehensive machine learning system for tennis underdog prediction and match analysis.

## Overview

Tennis One Set is an advanced tennis prediction system that specializes in identifying strong underdogs likely to win the second set in ATP and WTA singles tournaments. The system focuses on players ranked 50-300 and uses machine learning models to improve prediction accuracy.

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

- **Second Set Focus**: Specialized ML models for second set predictions
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