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
import pathlib

# This reverse proxy application is needed to span the small gaps when
# `cudaq-qpud` is shutting down and starting up again. This small reverse proxy
# allows the NVCF port (3030) to remain up while allowing the main `cudaq-qpud`
# application to restart if necessary.
PROXY_PORT = 3030
QPUD_PORT = 3031  # see `scripts/nvqc_launch.sh`

NUM_GPUS = 0
MPI_FOUND = False
WATCHDOG_TIMEOUT_SEC = 0
RUN_AS_NOBODY = False  # Expect this to be overridden to true for NVQC deployment
SUDO_FOUND = False
CUDAQ_SER_CODE_EXEC = False


def build_command_list(temp_file_name: str) -> list[str]:
    """
    Build the command essentially from right to left, pre-pending wrapper
    commands as necessary for this invocation.
    """
    current_script_path = os.path.abspath(__file__)
    json_req_path = os.path.join(os.path.dirname(current_script_path),
                                 'json_request_runner.py')
    cmd_list = [sys.executable, json_req_path, temp_file_name]
    if NUM_GPUS > 1 and MPI_FOUND:
        cmd_list = ['mpiexec', '--allow-run-as-root', '-np',
                    str(NUM_GPUS)] + cmd_list
        cmd_list += ['--use-mpi=1']  # `--use-mpi` must come at the end
    else:
        cmd_list += ['--use-mpi=0']  # `--use-mpi` must come at the end
    # The timeout must be inside the `su`/`sudo` commands in order to function.
    if WATCHDOG_TIMEOUT_SEC > 0:
        cmd_list = ['timeout', str(WATCHDOG_TIMEOUT_SEC)] + cmd_list
    if RUN_AS_NOBODY:
        cmd_list = ['su', '-s', '/bin/bash', 'nobody', '-c', ' '.join(cmd_list)]
        if SUDO_FOUND:
            cmd_list = ['sudo'] + cmd_list

    return cmd_list


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
            except Exception:
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
                    if CUDAQ_SER_CODE_EXEC:
                        result = {'status': 'uninitialized', 'errorMessage': ''}
                        with tempfile.NamedTemporaryFile(
                                dir=temp_dir, delete=False) as temp_file:
                            temp_file.write(request_data)
                            temp_file.flush()

                        # Make it world writable so that the `subprocess` can write
                        # the results to the file.
                        os.chmod(temp_file.name, 0o666)

                        # We also must get to a directory where "nobody" can see (in
                        # order to make MPI happy)
                        save_dir = os.getcwd()
                        os.chdir(pathlib.Path(temp_file.name).parent)
                        cmd_list = build_command_list(temp_file.name)
                        cmd_result = subprocess.run(cmd_list,
                                                    capture_output=False,
                                                    text=True)

                        with open(temp_file.name, 'rb') as fp:
                            result = json.load(fp)

                        if cmd_result.returncode == 124:
                            result = {
                                'status':
                                    'json_request_runner.py time out',
                                'errorMessage':
                                    'Timeout occurred during execution'
                            }

                        # Cleanup
                        os.chdir(save_dir)
                        if RUN_AS_NOBODY:
                            if SUDO_FOUND:
                                os.system('sudo pkill -9 -u nobody')
                            else:
                                os.system('pkill -9 -u nobody')
                        os.remove(temp_file.name)
                    else:
                        result = {
                            'status':
                                'Invalid Request',
                            'errorMessage':
                                'Server does not support serializedCodeExecutionContext at this time'
                        }

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
    SUDO_FOUND = (shutil.which('sudo') != None)
    WATCHDOG_TIMEOUT_SEC = int(
        os.environ.get('WATCHDOG_TIMEOUT_SEC', WATCHDOG_TIMEOUT_SEC))
    RUN_AS_NOBODY = int(os.environ.get('RUN_AS_NOBODY', 0)) > 0
    CUDAQ_SER_CODE_EXEC = int(
        os.environ.get('CUDAQ_SER_CODE_EXEC', CUDAQ_SER_CODE_EXEC)) > 0

    temp_dir = tempfile.gettempdir()
    if RUN_AS_NOBODY:
        temp_dir = os.path.join(temp_dir, 'nvqc_proxy')
        os.makedirs(temp_dir, exist_ok=True)
        os.chmod(temp_dir, 0o777)  # Allow "nobody" to write to this directory.

    Handler = Server
    with ThreadedHTTPServer(("", PROXY_PORT), Handler) as httpd:
        print("Serving at port", PROXY_PORT)
        print("Forward to port", QPUD_PORT)
        httpd.serve_forever()
