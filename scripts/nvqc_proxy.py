# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from http import HTTPStatus
from concurrent.futures import ProcessPoolExecutor
from cudaq import spin
import http.server
import json
import requests
import socketserver
import sys
import time
import base64
import py
import asyncio
import json
import ast
import subprocess
import pickle
import multiprocessing
from request_validator import RequestValidator

# This reverse proxy application is needed to span the small gaps when
# `cudaq-qpud` is shutting down and starting up again. This small reverse proxy
# allows the NVCF port (3030) to remain up while allowing the main `cudaq-qpud`
# application to restart if necessary.
PROXY_PORT = 22030
QPUD_PORT = 22031  # see `docker/build/cudaq.nvqc.Dockerfile`


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in a separate thread."""


class Server(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    default_request_version = 'HTTP/1.1'

    def __init__(self, *args, **kwargs):
        self.validator = RequestValidator()
        super().__init__(*args, **kwargs)

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

    def get_deserialized_dict(self, globals):
        deserialized_dict = {}

        for key, serialized_bytes in globals.items():
            try:
                deserialized_value = pickle.loads(serialized_bytes)
                deserialized_dict[key] = deserialized_value
            except (pickle.UnpicklingError, TypeError, Exception) as e:
                print(f"Error deserializing key '{key}': {e}")

        return deserialized_dict

    def validate_execution_context(self, source_code, deserialized_globals_dict):
        validation_context = {
            'source_code': source_code,
            'globals': deserialized_globals_dict
        }
        is_valid, validation_message = self.validator.validate_request(validation_context)
        if not is_valid:
            return False, {
                "status": "error",
                "error": validation_message
            }
        return True, None

    async def handle_job_request(self, request_json):
        serialized_ctx = request_json['serializedCodeExecutionContext']
        source_code = pickle.loads(base64.b64decode(serialized_ctx['source_code']))
        serialized_dict = pickle.loads(base64.b64decode(serialized_ctx['globals']))
        deserialized_globals_dict = self.get_deserialized_dict(serialized_dict)

        is_valid, validation_response = self.validate_execution_context(source_code, deserialized_globals_dict)
        if not is_valid:
            return validation_response

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, self.execute_code_in_subprocess, source_code, deserialized_globals_dict)

        return result

    @staticmethod
    def execute_code_in_subprocess(source_code, globals_dict):
        try:
            exec(source_code, globals_dict)
            result = {
                "status": "success",
                "executionContext" : {
                    "shots": 0,
                    "hasConditionalsOnMeasureResults": False,
                    "optResult": [
                        globals_dict['energy'],
                        globals_dict['params_at_energy']
                    ]
                }
            }
            return result
        except Exception as e:
            import traceback
            print("An error occured:")
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }

    def is_serialized_code_execution_request(self, request_json):
        return 'serializedCodeExecutionContext' in request_json and 'source_code' in request_json['serializedCodeExecutionContext'] and request_json['serializedCodeExecutionContext']['source_code'] != ''

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
                request_data = self.rfile.read(content_length)
                request_json = json.loads(request_data)
                
                if self.is_serialized_code_execution_request(request_json):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.handle_job_request(request_json))
                    loop.close()

                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'application/json')
                    message = json.dumps(result).encode('utf-8')
                    self.send_header('Content-Length', str(len(message)))
                    self.end_headers()
                    self.wfile.write(message)
                else:
                    res = requests.request(method=self.command,
                                        url=qpud_url + self.path,
                                        headers=self.headers,
                                        data=request_data)
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


if __name__ == "__main__":
    # By default, Linux uses the "fork" method where the child process inherits
    # a copy of the parent's memory space, including any condition variables and
    # locks that might be in use. If the parent process was holding a lock or
    # waiting on a condition variable at the time of the fork, this can lead to
    # deadlocks or other synchronization problems in the child process. Instead,
    # use the "spawn" method instead, where a fresh Python interpreter is
    # started for each new process.
    multiprocessing.set_start_method('spawn')

    Handler = Server
    with ThreadedHTTPServer(("", PROXY_PORT), Handler) as httpd:
        print("Serving at port", PROXY_PORT)
        print("Forward to port", QPUD_PORT)
        httpd.serve_forever()
