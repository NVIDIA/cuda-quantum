from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class QiboMockServer(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if self.path == '/api/jobs':
            # Create a job
            response = {
                'pid': 'job-123',
                'status': 'QUEUED'
            }
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

    def do_GET(self):
        if self.path.startswith('/api/jobs/job-123'):
            # Return job status and results
            response = {
                'pid': 'job-123',
                'status': 'success',
                'results': {
                    'counts': {
                        '00': 500,
                        '11': 500
                    }
                }
            }
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())

def startServer(port=62450):
    server_address = ('', port)
    httpd = HTTPServer(server_address, QiboMockServer)
    print(f'Starting mock server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    startServer()