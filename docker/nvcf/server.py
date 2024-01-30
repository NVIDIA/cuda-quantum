
from flask import Flask, jsonify, request
import fire
from typing import Union, List
import pathlib
import os
import json


class Server:
    def __init__(self, port: int = 3030):
        self.api = Flask(__name__)
        self.port = port
      
        @self.api.route('/job', methods=["POST"])
        def run():
            try:
                request_header = None
                asset_folder = None
                asset_id = None
                request_data = request.get_json()
                if hasattr(request, "headers"):
                    request_header = request.headers
                print("Request data:", request_data)
                print("Request header:", request_header)
                
                if request_header and request_header.get("NVCF-ASSET-DIR"):
                    asset_folder = request_header.get("NVCF-ASSET-DIR")
                
                if request_header and request_header.get("NVCF-FUNCTION-ASSET-IDS"):
                    asset_id = request_header.get("NVCF-FUNCTION-ASSET-IDS")
                    asset_id = asset_id.split(",")
                    if len(asset_id) == 1:
                        asset_id = asset_id[0]    
                    else:
                        asset_id = None
                
                if asset_folder and asset_id:
                    request_json_path = pathlib.Path(os.path.join(asset_folder, asset_id))
                    data = request_json_path.read_text()
                    print("Asset data:", data)
                    # Just replay the data back
                    data_json = json.loads(data)
                    #return json.loads(data)
                    return {"input": data_json}
                return {"input": request_data}
            
            except Exception as e:
                import traceback
                return {"error": traceback.format_exc()}

        @self.api.route('/', methods=["GET"])
        def status():
            return {"status": "OK"}
    
    def run(self):
        self.api.run(debug=False, host="0.0.0.0", port=self.port)
    
if __name__ == '__main__':
    fire.Fire(Server)
