# Tennis One Set - Test Suite Organization

## Test Structure

This test suite is organized into logical categories for better maintainability and clarity:

### 📁 Directory Structure

```
src/tests/
├── conftest.py              # Pytest configuration and fixtures
├── pytest.ini              # Pytest settings and markers
├── README.md               # This file
├── api/                    # API endpoint tests
│   ├── comprehensive_api_test.py
│   ├── test_api_integration_full.py
│   ├── test_api_tennis_integration.py
│   ├── test_live_odds_endpoints.py
│   └── test_odds_endpoints.py
├── integration/            # Integration tests (multiple components)
│   ├── ranking_integration_test_and_guide.py
│   ├── test_betting_pipeline.py
│   ├── test_complete_pipeline.py
│   ├── test_secure_filtering_integration.py
│   └── test_websocket_integration.py
├── unit/                   # Unit tests (single components)
│   ├── test_ml_simple.py
│   ├── test_ranking_api_methods.py
│   ├── test_ranking_filter_fix.py
│   └── test_server_connection.py
├── telegram/               # Telegram integration tests
│   ├── quick_telegram_test.py
│   ├── telegram_test_corrected.py
│   ├── telegram_test_fixed.py
│   ├── test_telegram_direct.py
│   ├── test_telegram_integration.py
│   └── verify_telegram_security.py
├── legacy/                 # Legacy/deprecated tests
│   ├── test_scheduled.py
│   └── test_scheduled_matches.py
├── demos/                  # Demo and example scripts
│   └── external_api_tennis_demo.py
└── verification/           # Security and compliance tests
    └── verify_tls_fixes.py
```

## Running Tests

### All Tests
```bash
# From project root
pytest src/tests/

# With verbose output
pytest -v src/tests/
```

### By Category
```bash
# API tests only
pytest src/tests/api/

# Integration tests only
pytest src/tests/integration/

# Unit tests only  
pytest src/tests/unit/

# Telegram tests only
pytest src/tests/telegram/
```

### By Markers
```bash
# Fast tests only
pytest -m "not slow" src/tests/

# External API tests (require network)
pytest -m external src/tests/

# Telegram-specific tests
pytest -m telegram src/tests/telegram/
```

## Test Categories

### 🔌 API Tests (`src/tests/api/`)
- **Purpose**: Test API endpoints and external integrations
- **Scope**: HTTP endpoints, request/response validation, API connectivity
- **Requirements**: Running Flask server, API keys for external services

### 🔗 Integration Tests (`src/tests/integration/`)
- **Purpose**: Test multiple components working together
- **Scope**: End-to-end workflows, data pipelines, system interactions
- **Requirements**: Full system setup, external dependencies

### 🧪 Unit Tests (`src/tests/unit/`)
- **Purpose**: Test individual functions and classes
- **Scope**: Isolated component testing, business logic validation
- **Requirements**: Minimal dependencies, fast execution

### 💬 Telegram Tests (`src/tests/telegram/`)
- **Purpose**: Test Telegram bot integration and notifications
- **Scope**: Bot commands, message formatting, security validation
- **Requirements**: Telegram bot token, chat ID configuration

### 📁 Legacy Tests (`src/tests/legacy/`)
- **Purpose**: Deprecated tests kept for reference
- **Scope**: Old scheduling system, deprecated features
- **Status**: Not actively maintained, may be removed

### 🎯 Demo Scripts (`src/tests/demos/`)
- **Purpose**: Example usage and demonstration scripts
- **Scope**: API showcases, tutorial examples
- **Usage**: Learning and development reference

### 🔒 Verification Tests (`src/tests/verification/`)
- **Purpose**: Security and compliance validation
- **Scope**: TLS configuration, security headers, vulnerability checks
- **Requirements**: Network access, security tools

## Configuration

### Environment Variables
Set these in your `.env` file for comprehensive testing:

```bash
# API Keys
API_TENNIS_KEY=your_api_tennis_key
RAPIDAPI_KEY=your_rapidapi_key
ODDS_API_KEY=your_odds_api_key

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Testing
FLASK_ENV=testing
DATABASE_URL=sqlite:///test.db
```

### Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.api` - API tests
- `@pytest.mark.telegram` - Telegram tests
- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.external` - Tests requiring external API access

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mocking**: Use mocks for external dependencies in unit tests
3. **Fixtures**: Leverage pytest fixtures for common setup
4. **Markers**: Tag tests appropriately for selective execution
5. **Documentation**: Include docstrings explaining test purpose

## Contributing

When adding new tests:

1. Place them in the appropriate category directory
2. Follow the naming convention: `test_*.py`
3. Add appropriate pytest markers
4. Include docstrings and comments
5. Ensure tests are independent and repeatable

## Cleanup Status

✅ **Completed**:
- Organized 19 test files into logical categories
- Created pytest configuration
- Added proper directory structure
- Documented test organization

🔄 **Next Steps**:
- Review and consolidate duplicate telegram tests
- Add more comprehensive unit test coverage
- Implement CI/CD test automation
- Add performance benchmarking tests