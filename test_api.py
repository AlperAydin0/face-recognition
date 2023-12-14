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
    api_url = "http://127.0.0.1:8000/process-images"  # Replace with your FastAPI endpoint URL

    # Replace these paths with the actual paths to your image files
    image_path1 = "data/my_face/1.jpg"
    image_path2 = "data/my_face/2.jpg"

    # Encode images to base64
    encoded_image1 = encode_image_to_base64(image_path1)
    encoded_image2 = encode_image_to_base64(image_path2)

    if not encoded_image1 or not encoded_image2:
        print("Error encoding images.")
        return

    # Prepare the request payload
    payload = {
        "image1": encoded_image1,
        "image2": encoded_image2
    }

    # Make the POST request to the FastAPI endpoint
    response = requests.post(api_url, json=payload)
    # Print the response
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.json())

if __name__ == "__main__":
    test_api()
