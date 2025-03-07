# Simple python utility to obtain token from Pasqal Cloud using PASQAL_USERNAME/PASQAL_PASSWORD
from pasqal_cloud import SDK
import os

sdk = SDK(
    username=os.environ.get("PASQAL_USERNAME"),
    password=os.environ.get("PASQAL_PASSWORD", None),
)

print(sdk.user_token())
