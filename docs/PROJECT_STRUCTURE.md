# Tennis One Set - Project Structure

## 🎾 Overview
A machine learning-driven tennis prediction system focused on identifying underdog outcomes in second sets for ATP and WTA matches.

## 📁 Directory Structure

```
Tennis_one_set/
├── 🔧 Core Application
│   ├── main.py                     # Main application entry point
│   ├── requirements.txt            # Python dependencies
│   └── .env.production.template    # Environment variables template
│
├── 📊 Source Code (src/)
│   ├── api/                        # Flask web API
│   │   ├── app.py                  # Flask application factory
│   │   ├── routes.py               # API endpoints
│   │   ├── middleware.py           # Security & rate limiting
│   │   └── dynamic_rankings_api.py # Live rankings integration
│   │
│   ├── ml/                         # Machine Learning modules
│   │   ├── enhanced_feature_engineering.py  # Advanced features
│   │   ├── bayesian_hyperparameter_optimizer.py  # Model optimization
│   │   ├── realtime_data_collector.py  # Live data pipeline
│   │   ├── dynamic_ensemble.py     # Ensemble methods
│   │   ├── lstm_sequential_model.py # Neural networks
│   │   └── enhanced_pipeline.py    # Complete ML pipeline
│   │
│   ├── data/                       # Data processing
│   │   ├── collectors/             # Data collection modules
│   │   ├── processors/             # Data processing utilities
│   │   └── validators/             # Data validation
│   │
│   ├── utils/                      # Utilities
│   │   ├── telegram_notification_system.py  # Notifications
│   │   ├── health_checker.py       # System monitoring
│   │   └── logger.py               # Logging utilities
│   │
│   ├── config/                     # Configuration
│   │   └── config.py               # Application settings
│   │
│   └── tests/                      # Test suite
│       ├── unit/                   # Unit tests
│       ├── integration/            # Integration tests
│       └── api/                    # API tests
│
├── 🎨 Frontend (static/ & templates/)
│   ├── static/
│   │   ├── css/                    # Stylesheets
│   │   ├── js/                     # JavaScript files
│   │   └── sounds/                 # Audio notifications
│   └── templates/                  # HTML templates
│
├── 📈 Data Storage
│   ├── cache/                      # API response cache
│   ├── logs/                       # Application logs
│   └── reports/                    # Generated reports
│
├── 🤖 ML Models
│   └── tennis_models/              # Trained model files
│
├── 🐳 Deployment
│   ├── Dockerfile                  # Container definition
│   ├── docker-compose.yml          # Multi-container setup
│   └── start_tennis_predictions.sh # Startup script
│
├── 📚 Documentation
│   ├── docs/                       # Comprehensive documentation
│   ├── README.md                   # Project overview
│   └── PROJECT_STRUCTURE.md        # This file
│
└── 🔧 Utilities
    ├── scripts/                    # Utility scripts
    ├── generate_performance_report.py  # Performance analysis
    └── migrate_real_data.py        # Data migration
```

## 🚀 Quick Start

### Development Mode (Recommended)
```bash
# 1. Activate virtual environment
source tennis_one_set_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python main.py
```

### Production Mode (Docker)
```bash
# 1. Stop any running instances
docker-compose down

# 2. Build and start
docker-compose up --build -d

# 3. Monitor logs
docker-compose logs -f
```

## 🎯 Key Features

- **🧠 Advanced ML Pipeline**: Multiple models with ensemble learning
- **📊 Real-time Data**: Live match and ranking data integration
- **🔄 Automated Predictions**: Every 30 minutes analysis cycle
- **📱 Web Dashboard**: Interactive betting analytics interface
- **📢 Telegram Notifications**: Automated alert system
- **🏥 Health Monitoring**: System status and API health checks

## 🔑 Environment Variables

Create `.env.production` file with:
```bash
API_TENNIS_KEY=your_api_tennis_key
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
FLASK_SECRET_KEY=your_secret_key
```

## 📊 System Access

- **Dashboard**: http://localhost:5001/
- **API Health**: http://localhost:5001/api/health
- **API Docs**: http://localhost:5001/api/docs

## 🎾 Focus Area

The system specializes in **second-set underdog predictions** for tennis matches, targeting players ranked 50-300 with high-accuracy ML models and real-time data analysis.# Tennis One Set - Project Structure

## 🎾 Overview
A machine learning-driven tennis prediction system focused on identifying underdog outcomes in second sets for ATP and WTA matches.

## 📁 Directory Structure

```
Tennis_one_set/
├── 🔧 Core Application
│   ├── main.py                     # Main application entry point
│   ├── requirements.txt            # Python dependencies
│   └── .env.production.template    # Environment variables template
│
├── 📊 Source Code (src/)
│   ├── api/                        # Flask web API
│   │   ├── app.py                  # Flask application factory
│   │   ├── routes.py               # API endpoints
│   │   ├── middleware.py           # Security & rate limiting
│   │   └── dynamic_rankings_api.py # Live rankings integration
│   │
│   ├── ml/                         # Machine Learning modules
│   │   ├── enhanced_feature_engineering.py  # Advanced features
│   │   ├── bayesian_hyperparameter_optimizer.py  # Model optimization
│   │   ├── realtime_data_collector.py  # Live data pipeline
│   │   ├── dynamic_ensemble.py     # Ensemble methods
│   │   ├── lstm_sequential_model.py # Neural networks
│   │   └── enhanced_pipeline.py    # Complete ML pipeline
│   │
│   ├── data/                       # Data processing
│   │   ├── collectors/             # Data collection modules
│   │   ├── processors/             # Data processing utilities
│   │   └── validators/             # Data validation
│   │
│   ├── utils/                      # Utilities
│   │   ├── telegram_notification_system.py  # Notifications
│   │   ├── health_checker.py       # System monitoring
│   │   └── logger.py               # Logging utilities
│   │
│   ├── config/                     # Configuration
│   │   └── config.py               # Application settings
│   │
│   └── tests/                      # Test suite
│       ├── unit/                   # Unit tests
│       ├── integration/            # Integration tests
│       └── api/                    # API tests
│
├── 🎨 Frontend (static/ & templates/)
│   ├── static/
│   │   ├── css/                    # Stylesheets
│   │   ├── js/                     # JavaScript files
│   │   └── sounds/                 # Audio notifications
│   └── templates/                  # HTML templates
│
├── 📈 Data Storage
│   ├── cache/                      # API response cache
│   ├── logs/                       # Application logs
│   └── reports/                    # Generated reports
│
├── 🤖 ML Models
│   └── tennis_models/              # Trained model files
│
├── 🐳 Deployment
│   ├── Dockerfile                  # Container definition
│   ├── docker-compose.yml          # Multi-container setup
│   └── start_tennis_predictions.sh # Startup script
│
├── 📚 Documentation
│   ├── docs/                       # Comprehensive documentation
│   ├── README.md                   # Project overview
│   └── PROJECT_STRUCTURE.md        # This file
│
└── 🔧 Utilities
    ├── scripts/                    # Utility scripts
    ├── generate_performance_report.py  # Performance analysis
    └── migrate_real_data.py        # Data migration
```

## 🚀 Quick Start

### Development Mode (Recommended)
```bash
# 1. Activate virtual environment
source tennis_one_set_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python main.py
```

### Production Mode (Docker)
```bash
# 1. Stop any running instances
docker-compose down

# 2. Build and start
docker-compose up --build -d

# 3. Monitor logs
docker-compose logs -f
```

## 🎯 Key Features

- **🧠 Advanced ML Pipeline**: Multiple models with ensemble learning
- **📊 Real-time Data**: Live match and ranking data integration
- **🔄 Automated Predictions**: Every 30 minutes analysis cycle
- **📱 Web Dashboard**: Interactive betting analytics interface
- **📢 Telegram Notifications**: Automated alert system
- **🏥 Health Monitoring**: System status and API health checks

## 🔑 Environment Variables

Create `.env.production` file with:
```bash
API_TENNIS_KEY=your_api_tennis_key
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
FLASK_SECRET_KEY=your_secret_key
```

## 📊 System Access

- **Dashboard**: http://localhost:5001/
- **API Health**: http://localhost:5001/api/health
- **API Docs**: http://localhost:5001/api/docs

## 🎾 Focus Area

The system specializes in **second-set underdog predictions** for tennis matches, targeting players ranked 50-300 with high-accuracy ML models and real-time data analysis.