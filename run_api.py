#!/usr/bin/env python3
"""
Launch script for FastAPI Video Interview System
"""

import subprocess
import sys
import os

def main():
    """Launch the FastAPI server."""
    print("🚀 Starting FastAPI Video Interview System...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Server URL: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Change to the directory containing the app
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # Launch FastAPI with Uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())