import subprocess
import sys
import os
import time
import signal

def main():
    print("🚀 Starting InsightFlow Services...")
    
    # Paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "landing-page")
    
    if not os.path.exists(frontend_dir):
        print("❌ Error: 'landing-page' directory not found.")
        sys.exit(1)

    # Start Streamlit App
    print("📈 Starting Streamlit backend on port 8501...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"],
        cwd=root_dir
    )

    # Start React Frontend
    print("💻 Starting React frontend on port 5173...")
    # On Windows, we need shell=True for npm commands to resolve correctly
    is_windows = sys.platform == "win32"
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=is_windows
    )

    print("\n✅ All services started successfully!")
    print("🌐 Landing Page: http://localhost:5173")
    print("⚙️  Streamlit App: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services.")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
        # Terminate processes
        streamlit_process.terminate()
        frontend_process.terminate()
        
        # Wait for them to close
        streamlit_process.wait()
        frontend_process.wait()
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
