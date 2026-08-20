# helper to replace header
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=30,
)
