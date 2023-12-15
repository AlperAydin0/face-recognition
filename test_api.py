import requests
import base64

def encode_image_to_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")
            return encoded_image
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def test_api():
    api_url = "http://127.0.0.1:8000/face-similarity"

    image_path1 = "data/images/3.jpg"
    image_path2 = "data/images/4.jpg"

    encoded_image1 = encode_image_to_base64(image_path1)
    encoded_image2 = encode_image_to_base64(image_path2)

    if not encoded_image1 or not encoded_image2:
        print("Error encoding images.")
        return

    payload = {
        "image1": encoded_image1,
        "image2": encoded_image2
    }

    # Make the POST request to the FastAPI endpoint
    response = requests.post(api_url, json=payload, headers={'api-key':"88f407bcdfc941c7b2e9aff1716e89b9"})
    # Print the response
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.json())

if __name__ == "__main__":
    test_api()
