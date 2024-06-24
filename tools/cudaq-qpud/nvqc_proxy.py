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
import json
import subprocess
import os
import tempfile
import shutil

# This reverse proxy application is needed to span the small gaps when
# `cudaq-qpud` is shutting down and starting up again. This small reverse proxy
# allows the NVCF port (3030) to remain up while allowing the main `cudaq-qpud`
# application to restart if necessary.
PROXY_PORT = 3030
QPUD_PORT = 3031  # see `docker/build/cudaq.nvqc.Dockerfile`

NUM_GPUS = 0
MPI_FOUND = False


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

    def is_serialized_code_execution_request(self, request_json):
        return 'serializedCodeExecutionContext' in request_json and 'source_code' in request_json[
            'serializedCodeExecutionContext'] and request_json[
                'serializedCodeExecutionContext']['source_code'] != ''

    def write_asset_if_necessary(self, message):
        """
        If the output message is too large, and if the proxy is servicing NVCF
        requests, then write the original message to a file and modify the
        outgoing message to reference that new file.
        """
        if 'NVCF-MAX-RESPONSE-SIZE-BYTES' in self.headers:
            max_response_len = int(self.headers['NVCF-MAX-RESPONSE-SIZE-BYTES'])
            if len(message) > max_response_len:
                try:
                    outputDir = self.headers['NVCF-LARGE-OUTPUT-DIR']
                    reqId = self.headers['NVCF-REQID']
                    resultFile = f'{outputDir}/{reqId}_result.json'
                    with open(resultFile, 'wb') as fp:
                        fp.write(message)
                        fp.flush()
                    result = {'resultFile': resultFile}
                    message = json.dumps(result).encode('utf-8')
                except Exception as e:
                    result = {
                        'status': 'Exception during output processing',
                        'errorMessage': str(e)
                    }
                    message = json.dumps(result).encode('utf-8')
        return message

    def read_asset_if_necessary(self, request_data):
        """
        If there is an asset ID in the headers, replace the incoming message
        with the contents of a file read from disk.
        """
        asset_id = self.headers.get('NVCF-FUNCTION-ASSET-IDS', '')
        if len(asset_id) > 0:
            try:
                asset_dir = self.headers['NVCF-ASSET-DIR']
                filename = f'{asset_dir}/{asset_id}'
                with open(filename, 'rb') as f:
                    request_data = f.read()
            except Exception as e:
                # If something failed, simply forward the original message
                pass
        return request_data

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
                # Look for any asset references in the job request. If one
                # exists, then that means the request is actually in a file.
                request_data = self.rfile.read(content_length)
                request_data = self.read_asset_if_necessary(request_data)
                request_json = json.loads(request_data)

                if self.is_serialized_code_execution_request(request_json):
                    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                        temp_file.write(request_data)
                        temp_file.flush()
                        current_script_path = os.path.abspath(__file__)
                        json_req_path = os.path.join(
                            os.path.dirname(current_script_path),
                            'json_request_runner.py')
                        cmd_list = [
                            sys.executable, json_req_path, temp_file.name
                        ]
                        if NUM_GPUS > 1 and MPI_FOUND:
                            cmd_list = [
                                'mpiexec', '--allow-run-as-root', '-np',
                                str(NUM_GPUS)
                            ] + cmd_list
                            os.environ['CUDAQ_USE_MPI'] = '1'
                        else:
                            os.environ['CUDAQ_USE_MPI'] = '0'
                        cmd_result = subprocess.run(cmd_list,
                                                    capture_output=True,
                                                    text=True)
                        last_line = cmd_result.stdout.strip().split('\n')[-1]
                        result = json.loads(last_line)

                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'application/json')
                    message = json.dumps(result).encode('utf-8')
                    message = self.write_asset_if_necessary(message)
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
    try:
        NUM_GPUS = int(subprocess.getoutput('nvidia-smi --list-gpus | wc -l'))
        if 'NUM_GPUS' in os.environ:
            NUM_GPUS = min(NUM_GPUS, int(os.environ['NUM_GPUS']))
    except:
        NUM_GPUS = 0
    MPI_FOUND = (shutil.which('mpiexec') != None)
    Handler = Server
    with ThreadedHTTPServer(("", PROXY_PORT), Handler) as httpd:
        print("Serving at port", PROXY_PORT)
        print("Forward to port", QPUD_PORT)
        httpd.serve_forever()
