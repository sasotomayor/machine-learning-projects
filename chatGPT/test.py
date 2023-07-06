import requests

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer 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'
}

data = {
    'model': 'gpt-3.5-turbo',
    'messages': [
        {'role': 'user', 'content': 'What is the weather like today?'}
    ]
}

response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
print(response.json())
