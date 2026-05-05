import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

load_dotenv()

credential = DefaultAzureCredential()
client = SecretClient(
    vault_url=os.getenv("AZURE_KEYVAULT_URL"),
    credential=credential
)
secret = client.get_secret("STORAGE-CONNECTION-STRING")
print("Key Vault connection: OK")
print(f"Secret length: {len(secret.value)} chars")