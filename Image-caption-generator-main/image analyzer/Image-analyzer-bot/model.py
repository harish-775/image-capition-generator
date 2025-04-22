from flask import Flask, request, jsonify, render_template
from PIL import Image
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import requests

# For testing an external API, use a valid URL
try:
    response = requests.get("https://valid-api-url.com", verify=False)  # Disable SSL verification for testing
    print(response.status_code)
except Exception as e:
    print(f"Error: {e}")

app = Flask(__name__)

# Replace this with your actual API key from Google Cloud
api_key = "AIzaSyBVec6inSuyB7rE4qOPdskMM_UTWQ525GM"

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Define a temporary directory within the project
tmp_dir = os.path.join(os.getcwd(), 'temp')
os.makedirs(tmp_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze():
    print("Received POST request")

    # Check if an image is uploaded
    if 'image' not in request.files:
        print("No image found in request")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    prompt = request.form.get('prompt')
    print(f"Prompt received: {prompt}")

    # Check if prompt is missing
    if not prompt:
        print("Prompt is missing")
        return jsonify({"error": "Prompt is missing"}), 400

    # Check if image filename is empty
    if image.filename == '':
        print("Empty filename")
        return jsonify({"error": "No selected image"}), 400

    image_filename = secure_filename(image.filename)
    image_path = os.path.join(tmp_dir, image_filename)
    image.save(image_path)
    print(f"Saved image to {image_path}")

    try:
        # Upload image to Gemini model
        uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")
        print("Image uploaded to Gemini")

        # Create a chat session with the uploaded image and prompt
        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [uploaded_file, prompt],
            }]
        )

        response = chat_session.send_message(prompt)
        print("Raw response:", response)
        print("Response type:", type(response))

        # Handle response based on its type or structure
        if isinstance(response, dict):  # Assuming the response is a dictionary
            result = response.get('text', 'No text found in response')
        else:
            result = str(response)  # Fallback to string representation

        print("Final result:", result)

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"result": result})

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

if __name__ == "__main__":
    app.run(debug=True, port=5656)
