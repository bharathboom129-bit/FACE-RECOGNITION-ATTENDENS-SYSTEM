import os
import sys
import webbrowser
import threading
import time

from app_web_full import app

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SMART CLASSROOM FULL-STACK WEB APPLICATION")
    print("   Running on: http://127.0.0.1:5000")
    print("=" * 70)

    # Automatically open browser
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host="127.0.0.1", port=5000, debug=False)
