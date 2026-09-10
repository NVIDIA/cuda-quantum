#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Serve the trace viewer so it can be opened from outside the container.

Binds 0.0.0.0 rather than localhost, which is what makes the port reachable
from the host. Serves this directory, so `viewer.html` and any trace JSON
written next to it are both available:

    python3 serve.py --port 8765
    # then, on the host:
    #   http://localhost:8765/viewer.html                  (embedded sample)
    #   http://localhost:8765/viewer.html?trace=my.json    (a trace beside it)

Use --dir to serve traces written elsewhere; `viewer.html` is copied in if it
isn't already there.
"""

import argparse
import functools
import http.server
import os
import shutil
import socketserver

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):

    def end_headers(self):
        # Traces are rewritten in place while the server runs; never cache.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print("  %s - %s" % (self.address_string(), fmt % args), flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dir", default=HERE,
                   help="directory to serve (default: the viewer's own)")
    p.add_argument("--bind", default="0.0.0.0",
                   help="interface to bind; 0.0.0.0 reaches outside the container")
    args = p.parse_args()

    root = os.path.abspath(args.dir)
    viewer = os.path.join(root, "viewer.html")
    if not os.path.exists(viewer):
        shutil.copy(os.path.join(HERE, "viewer.html"), viewer)
        print(f"copied viewer.html into {root}")

    traces = sorted(f for f in os.listdir(root) if f.endswith(".json"))

    handler = functools.partial(Handler, directory=root)
    with Server((args.bind, args.port), handler) as httpd:
        print(f"serving {root} on {args.bind}:{args.port}\n")
        print("  open from the host:")
        print(f"    http://localhost:{args.port}/viewer.html")
        for t in traces:
            print(f"    http://localhost:{args.port}/viewer.html?trace={t}")
        print("\n  If the host cannot reach it, the port is not forwarded.")
        print(f"  VS Code dev container: PORTS panel -> Forward a Port -> {args.port},")
        print(f"  or add \"forwardPorts\": [{args.port}] to devcontainer.json and rebuild.")
        print(f"  Plain docker: run the container with -p {args.port}:{args.port}.")
        print("  (The 172.x container address is not routable from a WSL2 host.)")
        print("\nCtrl-C to stop.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
