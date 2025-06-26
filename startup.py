#!/usr/bin/env python3
"""
🎾 Tennis Prediction System - Easy Startup Script
Launches the web dashboard and backend system
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask-cors', 'pandas', 'numpy', 
        'scikit-learn', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print(f"\n💡 Install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def create_project_structure():
    """Create necessary directories and files"""
    directories = [
        'templates',
        'static',
        'tennis_data_enhanced',
        'tennis_models',
        'betting_data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def save_dashboard_html():
    """Save the dashboard HTML to templates directory"""
    html_content = '''<!-- The dashboard HTML content would go here -->
<!-- For now, this is a placeholder. The full HTML is in the artifacts above -->
<!DOCTYPE html>
<html>
<head>
    <title>Tennis Dashboard</title>
</head>
<body>
    <h1>Tennis Prediction Dashboard</h1>
    <p>Please use the standalone HTML file from the artifacts above.</p>
    <p>The Flask backend will serve API endpoints at /api/*</p>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_content)
    
    print("📄 Dashboard template created")

def start_system():
    """Start the tennis prediction system"""
    print("🎾 TENNIS PREDICTION SYSTEM STARTUP")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("✅ All dependencies found")
    
    # Create project structure
    print("\n📁 Setting up project structure...")
    create_project_structure()
    save_dashboard_html()
    
    print("\n🚀 Starting system components...")
    
    # Check if Flask backend exists
    if not os.path.exists('tennis_web_backend.py'):
        print("⚠️ Backend file 'tennis_web_backend.py' not found")
        print("💡 Please save the Flask backend code from above as 'tennis_web_backend.py'")
        return False
    
    try:
        # Start Flask backend
        print("🔄 Starting Flask backend...")
        backend_process = subprocess.Popen([
            sys.executable, 'tennis_web_backend.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if server is running
        if backend_process.poll() is None:
            print("✅ Backend server started successfully")
            
            # Open browser
            print("🌐 Opening web dashboard...")
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
            
            print("\n" + "=" * 50)
            print("🎉 TENNIS SYSTEM READY!")
            print("=" * 50)
            print("🌐 Dashboard: http://localhost:5000")
            print("📊 API Endpoints:")
            print("   • /api/matches - Get upcoming matches")
            print("   • /api/stats - System statistics")
            print("   • /api/refresh - Refresh data")
            print("\n⏹️ Press Ctrl+C to stop the system")
            print("=" * 50)
            
            # Keep the script running
            try:
                backend_process.wait()
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down system...")
                backend_process.terminate()
                backend_process.wait()
                print("✅ System stopped")
                
        else:
            print("❌ Failed to start backend server")
            stdout, stderr = backend_process.communicate()
            print(f"Error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return False
    
    return True

def show_setup_instructions():
    """Show setup instructions"""
    print("""
🎾 TENNIS PREDICTION SYSTEM SETUP
================================

To get started:

1. 📦 SAVE THE FILES:
   • Save the Flask backend code as 'tennis_web_backend.py'
   • Save the HTML dashboard as 'tennis_dashboard.html'
   • Make sure your tennis prediction files are in the same directory

2. 📋 INSTALL DEPENDENCIES:
   pip install flask flask-cors pandas numpy scikit-learn requests

3. 🚀 RUN THE SYSTEM:
   python startup.py

4. 🌐 OPEN DASHBOARD:
   http://localhost:5000

📁 PROJECT STRUCTURE:
tennis-prediction/
├── tennis_web_backend.py      # Flask API server
├── tennis_dashboard.html      # Web dashboard
├── enhanced_data_collector.py # Your data collector
├── enhanced_predictor.py      # Your ML models
├── enhanced_betting_system.py # Your betting system
├── startup.py                 # This startup script
├── templates/                 # Flask templates
├── tennis_data_enhanced/      # Data storage
├── tennis_models/             # Trained models
└── betting_data/              # Betting history

🔧 CUSTOMIZATION:
• Edit API endpoints in tennis_web_backend.py
• Customize dashboard styling in the HTML file
• Adjust prediction logic in your existing modules
• Configure API keys and data sources

💡 TIPS:
• Start with the demo data to test the interface
• Gradually integrate your real data sources
• Monitor the console for any errors
• Use browser developer tools for debugging

🆘 TROUBLESHOOTING:
• Port 5000 already in use? Change port in the Flask app
• API errors? Check your tennis prediction modules
• No matches showing? Check the data collection logic
• CORS issues? Make sure flask-cors is installed

Ready to predict some tennis matches! 🎾
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Prediction System Startup')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--check', action='store_true', help='Check system dependencies')
    parser.add_argument('--install', action='store_true', help='Install missing dependencies')
    
    args = parser.parse_args()
    
    if args.setup:
        show_setup_instructions()
    elif args.check:
        print("🔍 Checking system dependencies...")
        if check_dependencies():
            print("✅ All dependencies are installed!")
        else:
            print("❌ Some dependencies are missing. Run with --install to fix.")
    elif args.install:
        print("📦 Installing missing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'flask', 'flask-cors', 'pandas', 'numpy', 
                'scikit-learn', 'requests', 'matplotlib', 'seaborn'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
    else:
        # Default: start the system
        success = start_system()
        if not success:
            print("\n💡 Run 'python startup.py --setup' for detailed instructions")
            sys.exit(1)