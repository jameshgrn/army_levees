
def post_request(url, id_number, is_json=True):
    headers = {
        'accept': 'application/json' if is_json else 'application/octet-stream',
        'Content-Type': 'application/json'
    }
    data = json.dumps([id_number])
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        if is_json:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text or f"Status code: {response.status_code}"
        else:
            # Handle binary response (file download)
            try:
                with open(f'{id_number}_geometry.zip', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                return "File downloaded successfully."
            except IOError as e:
                return f"File write error: {e}"
    else:
        return f"Status code: {response.status_code}, Reason: {response.reason}"
