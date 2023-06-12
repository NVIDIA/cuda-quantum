import sys
import json
from datetime import datetime


def write_a_mock_tokens_file(tokens_file_path):
    # Taken form the cortex-cli
    tokens_data = {
        "access_token": "good_access_token",
    }

    json_str = json.dumps(tokens_data)
    with open(tokens_file_path, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    tokens_file_path = sys.argv[1]
    write_a_mock_tokens_file(tokens_file_path)
