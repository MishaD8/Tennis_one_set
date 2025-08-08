#!/usr/bin/env python3
"""
🛠️ Database Setup and Installation Script
Sets up PostgreSQL for tennis prediction system
"""

import os
import subprocess
import sys
from database_models import DatabaseManager

def install_postgresql():
    """Install PostgreSQL if not already installed"""
    print("🔧 Checking PostgreSQL installation...")
    
    try:
        # Check if PostgreSQL is installed
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PostgreSQL already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("📦 Installing PostgreSQL...")
    
    # Installation commands for different systems
    if sys.platform.startswith('linux'):
        try:
            # Ubuntu/Debian
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'postgresql', 'postgresql-contrib'], check=True)
            print("✅ PostgreSQL installed on Linux")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PostgreSQL on Linux")
            return False
    
    elif sys.platform == 'darwin':
        try:
            # macOS with Homebrew
            subprocess.run(['brew', 'install', 'postgresql'], check=True)
            subprocess.run(['brew', 'services', 'start', 'postgresql'], check=True)
            print("✅ PostgreSQL installed on macOS")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PostgreSQL on macOS")
            return False
    
    else:
        print("❌ Unsupported platform for automatic PostgreSQL installation")
        print("Please install PostgreSQL manually for your system")
        return False

def setup_database():
    """Create database and user for tennis predictions"""
    print("🗄️ Setting up tennis predictions database...")
    
    # Database setup SQL commands
    setup_commands = [
        "CREATE DATABASE tennis_predictions;",
        "CREATE USER tennis_user WITH PASSWORD 'secure_password_here';",
        "GRANT ALL PRIVILEGES ON DATABASE tennis_predictions TO tennis_user;",
        "ALTER USER tennis_user CREATEDB;"
    ]
    
    try:
        for command in setup_commands:
            subprocess.run([
                'psql', '-U', 'postgres', '-c', command
            ], check=True, capture_output=True)
        
        print("✅ Database and user created successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Database setup failed: {e}")
        print("You may need to run this manually as postgres user:")
        for cmd in setup_commands:
            print(f"  {cmd}")
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("🐍 Installing Python dependencies...")
    
    dependencies = [
        'sqlalchemy',
        'psycopg2-binary',  # PostgreSQL adapter for Python
        'pandas',
        'python-dotenv'     # For environment variables
    ]
    
    try:
        for package in dependencies:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        
        print("✅ Python dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def test_database_connection():
    """Test connection to PostgreSQL database"""
    print("🔗 Testing database connection...")
    
    try:
        db_manager = DatabaseManager()
        if db_manager.test_connection():
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def create_env_file():
    """Create .env file from template"""
    print("📝 Creating environment configuration...")
    
    env_content = """# Tennis Prediction System Environment Variables
DATABASE_URL=postgresql://tennis_user:tennis_pass@localhost:5432/tennis_predictions
ODDS_API_KEY=your_odds_api_key_here
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development
LOG_LEVEL=INFO
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️ Remember to update ODDS_API_KEY in .env file")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Run complete database setup"""
    print("🚀 Tennis Prediction System Database Setup")
    print("=" * 50)
    
    success = True
    
    # Step 1: Install PostgreSQL
    if not install_postgresql():
        success = False
    
    # Step 2: Install Python dependencies
    if not install_python_dependencies():
        success = False
    
    # Step 3: Setup database
    if success and not setup_database():
        success = False
    
    # Step 4: Create environment file
    if not create_env_file():
        success = False
    
    # Step 5: Test connection
    if success and not test_database_connection():
        success = False
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with correct API keys")
        print("2. Run database migration: python database_migration.py")
        print("3. Update your code to use the new PostgreSQL logger")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and fix issues manually")

if __name__ == "__main__":
    main()