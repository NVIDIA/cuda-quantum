# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from http import HTTPStatus
import http.server
import json
import requests
import socketserver
import sys
import time

# This reverse proxy application is needed to span the small gaps when
# `cudaq-qpud` is shutting down and starting up again. This small reverse proxy
# allows the NVCF port (3030) to remain up while allowing the main `cudaq-qpud`
# application to restart if necessary.
PROXY_PORT = 3030
QPUD_PORT = 3031  # see `docker/build/cudaq.nvqc.Dockerfile`


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in a separate thread."""


class Server(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    default_request_version = 'HTTP/1.1'

    # Override this function because we seem to be getting a lot of
    # ConnectionResetError exceptions in the health monitoring endpoint,
    # producing lots of ugly stack traces in the logs. Hopefully this will
    # reduce them.
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ConnectionResetError as e:
            if self.path != '/':
                print(f"Connection was reset by peer: {e}")
        except Exception as e:
            print(f"Unhandled exception: {e}")

    def log_message(self, format, *args):
        # Don't log the health endpoint queries
        if len(args) > 0 and args[0] != "GET / HTTP/1.1":
            super().log_message(format, *args)

    def do_GET(self):
        # Allow the proxy to automatically handle the health endpoint. The proxy
        # will exit if the application's /job endpoint is down.
        if self.path == '/':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            message = json.dumps({"status": "OK"}).encode('utf-8')
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        if self.path == '/job':
            qpud_up = False
            retries = 0
            qpud_url = 'http://localhost:' + str(QPUD_PORT)
            while (not qpud_up):
                try:
                    ping_response = requests.get(qpud_url)
                    qpud_up = (ping_response.status_code == HTTPStatus.OK)
                except:
                    qpud_up = False
                if not qpud_up:
                    retries += 1
                    if retries > 100:
                        print("PROXY EXIT: TOO MANY RETRIES!")
                        sys.exit()
                    print(
                        "Main application is down, retrying (retry_count = {})..."
                        .format(retries))
                    time.sleep(0.1)

            content_length = int(self.headers['Content-Length'])
            if content_length:
                res = requests.request(method=self.command,
                                       url=qpud_url + self.path,
                                       headers=self.headers,
                                       data=self.rfile.read(content_length))
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/json')
                message = json.dumps(res.json()).encode('utf-8')
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Length", "0")
                self.end_headers()
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.end_headers()


Handler = Server
with ThreadedHTTPServer(("", PROXY_PORT), Handler) as httpd:
    print("Serving at port", PROXY_PORT)
    print("Forward to port", QPUD_PORT)
    httpd.serve_forever()
