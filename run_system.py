import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from student_monitor import StudentFocusMonitor
import os

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "dashboard"), **kwargs)

def run_monitor():
    monitor = StudentFocusMonitor(log_interval=5)
    monitor.run()

def run_server():
    PORT = 8000
    with ThreadingHTTPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Serving dashboard at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start the monitor in a separate thread
    monitor_thread = threading.Thread(target=run_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Start the web server in another thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Open the browser
    webbrowser.open_new_tab("http://localhost:8000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")