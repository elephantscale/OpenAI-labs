import os
from dotenv import load_dotenv
import json
import base64
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

def authenticate():
    #Load .env
    load_dotenv()
    #DLAI Custom Key
    return "DLAI_CREDENTIALS", "DLAI_PROJECT", "gs://gcp-sc2-rlhf"
    
    #Decode key and store in .JSON
    SERVICE_ACCOUNT_KEY_STRING_B64 = os.getenv('SERVICE_ACCOUNT_KEY')
    SERVICE_ACCOUNT_KEY_BYTES_B64 = SERVICE_ACCOUNT_KEY_STRING_B64.encode("ascii")
    SERVICE_ACCOUNT_KEY_STRING_BYTES = base64.b64decode(SERVICE_ACCOUNT_KEY_BYTES_B64)
    SERVICE_ACCOUNT_KEY_STRING = SERVICE_ACCOUNT_KEY_STRING_BYTES.decode("ascii")

    SERVICE_ACCOUNT_KEY = json.loads(SERVICE_ACCOUNT_KEY_STRING)


    # Create credentials based on key from service account
    # Make sure your account has the roles listed in the Google Cloud Setup section
    credentials = Credentials.from_service_account_info(
        SERVICE_ACCOUNT_KEY,
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    if credentials.expired:
        credentials.refresh(Request())
    
    #Set project ID according to environment variable    
    PROJECT_ID = os.getenv('PROJECT_ID')
    STAGING_BUCKET = os.getenv('STAGING_BUCKET')# 'gs://gcp-sc2-rlhf-staging'
    
    return credentials, PROJECT_ID, STAGING_BUCKET


def print_d(d, indent=0):
    for key, val in d.items():
        indentation = "  " * indent
        print(f"{indentation}" + "-" * 50)
        print(f"{indentation}key:{key}\n")
        if isinstance(val, dict):
            print(f"{indentation}val")
            print_d(val, indent=indent + 1)
        else:
            print(f"{indentation}val:{val}")

