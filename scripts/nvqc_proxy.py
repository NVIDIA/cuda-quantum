# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import socket
from threading import Thread
import queue
import time

# This reverse proxy application is needed to span the small gaps when
# `cudaq-qpud` is shutting down and starting up again. This small reverse proxy
# allows the NVCF port (3030) to remain up while allowing the main `cudaq-qpud`
# application to restart if necessary.

# Queue for storing requests
request_queue = queue.Queue()

# NVCF max payload is ~250KB. Round up to 1MB.
MAX_SIZE_BYTES = 1000000
NVCF_PORT = 3030
CUDAQ_QPUD_PORT = 3031  # see `docker/build/cudaq.nvqc.Dockerfile`


# Handle incoming client connections and queue their requests
def handle_client(client_socket):
    request = client_socket.recv(MAX_SIZE_BYTES)
    request_queue.put((client_socket, request))


# Forward requests from the queue to `cudaq-qpud`
def forward_requests_to_application():
    retry_count = 0
    while True:
        if not request_queue.empty():
            client_socket, request = request_queue.get(block=True, timeout=None)
            try:
                with socket.create_connection(
                    ("localhost", CUDAQ_QPUD_PORT)) as app_socket:
                    app_socket.sendall(request)
                    response = app_socket.recv(MAX_SIZE_BYTES)
                    client_socket.sendall(response)
                    client_socket.close()
                    retry_count = 0
            except ConnectionRefusedError:
                print(
                    "Main application is down, retrying (retry_count = {})...".
                    format(retry_count))
                retry_count += 1
                request_queue.put(
                    (client_socket,
                     request))  # Put back the request in the queue
        time.sleep(0.1)  # Throttle the retries


# Listen for incoming connections
def start_proxy_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("0.0.0.0", NVCF_PORT))
        server_socket.listen(5)  # max backlog
        print("Proxy server listening on port {}...".format(NVCF_PORT))

        # Start a thread to forward requests from the queue to the application
        Thread(target=forward_requests_to_application, daemon=True).start()

        while True:
            client_socket, addr = server_socket.accept()
            Thread(target=handle_client, args=(client_socket,)).start()


if __name__ == "__main__":
    start_proxy_server()
