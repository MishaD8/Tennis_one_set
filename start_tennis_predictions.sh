#!/bin/bash

# 🎾 Tennis Prediction Service Startup Script
# Starts the automated tennis prediction service that sends Telegram notifications

echo "🎾 TENNIS PREDICTION SERVICE"
echo "============================="

# Check if we're in the right directory
if [ ! -f "automated_tennis_prediction_service.py" ]; then
    echo "❌ Error: automated_tennis_prediction_service.py not found"
    echo "Please run this script from the Tennis_one_set directory"
    exit 1
fi

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3"
    exit 1
fi

# Check if virtual environment exists
if [ -d "tennis_one_set_env" ]; then
    echo "🔧 Activating virtual environment..."
    source tennis_one_set_env/bin/activate
fi

echo "🚀 Starting Tennis Prediction Service..."
echo ""
echo "📱 Telegram notifications will be sent for underdog opportunities"
echo "🎯 Target: ATP/WTA singles, ranks 10-300, >55% confidence"
echo "⏰ Checks every 30 minutes for new matches"
echo ""
echo "📋 To stop: Press Ctrl+C"
echo "📊 Logs: automated_tennis_predictions.log"
echo ""
echo "Starting service..."

# Start the service
python3 automated_tennis_prediction_service.py

echo ""
echo "🛑 Tennis Prediction Service stopped"