#!/usr/bin/env python3
"""
Start the FastAPI Dashboard with WebSockets
Real-time updates, no polling, instant communication
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if FastAPI requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import aiohttp
        print("✅ FastAPI dashboard requirements found")
        return True
    except ImportError as e:
        print(f"❌ Missing requirements: {e}")
        print("Installing FastAPI dashboard requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "aiohttp"])
            print("✅ FastAPI dashboard requirements installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def start_dashboard():
    """Start the FastAPI dashboard"""
    print("🚀 Starting Miku Bot Dashboard (FastAPI + WebSockets)")
    print("📊 Dashboard will be available at: http://localhost:5002")
    print("🔌 WebSocket endpoint: ws://localhost:5002/ws")
    print("📡 Bot API endpoint: http://localhost:5002/api/bot/status")
    print("🤖 Make sure the bot is running for full functionality")
    print("=" * 60)
    
    try:
        # Import and run dashboard
        from dashboard_fastapi import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=5002,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Try running as administrator or use a different port")

if __name__ == "__main__":
    if check_requirements():
        start_dashboard()
    else:
        print("❌ Cannot start dashboard without required packages")
        sys.exit(1)
