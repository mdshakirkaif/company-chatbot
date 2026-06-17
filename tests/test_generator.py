import requests
YOUR_MODAL_URL='https://mdshakirkaif--company-policy-bot-api.modal.run'
response = requests.post(
    YOUR_MODAL_URL,
    json={"prompt": "Who is the CEO of Microsoft?"}
)

print(response.json())