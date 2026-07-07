# api/index.py
from http.server import BaseHTTPRequestHandler
from io import BytesIO
import json
import os
import sys
import threading
import time

# استيراد البوت
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "running",
            "bot": "Falcon AI Pro",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())
