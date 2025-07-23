import io
import PIL.Image
from flask import Flask, redirect, request, Response
from flask_cors import CORS # Import CORS

import google.generativeai as genai

# Configure Gemini API key
# IMPORTANT: For production, do not hardcode API keys. Use environment variables.
genai.configure(api_key="AIzaSyBJqnwsEDyhXkD73L04O7wxYHIxfCJLETU")

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing requests from your Next.js frontend

@app.route('/', methods=['GET'])
def index():
    """
    Redirects the root URL of the Flask server to the Next.js frontend application.
    This ensures that if someone accesses the Flask server directly, they are
    sent to the main user interface.
    """
    return redirect("https://stonesweb.in/tools/genderage")

@app.route('/gemini_upload', methods=['POST'])
def gemini_upload():
    """
    Handles image uploads from the frontend, processes them using the Gemini API,
    and returns a structured text response.
    """
    # Check if 'gemini_image' file is present in the request
    if 'gemini_image' not in request.files:
        return "No Gemini image uploaded", 400

    file = request.files['gemini_image']

    # Check if a file was actually selected
    if file.filename == '':
        return "No Gemini image selected", 400

    image_bytes = file.read() # Read the image data from the uploaded file

    try:
        # Initialize the Gemini Generative Model
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Open the image bytes as a PIL Image
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Define the prompt for the Gemini model
        prompt = (
            "Give me your **best guess** of the person's age and gender from this image. "
            "The age must be a blunt, precise 3-year range only (e.g., 20-23, 30-33 — nothing fuzzy like 'young adult'). "
            "Gender must be either MALE or FEMALE. Be funny and direct, like Grok — no sugarcoating, just say it like it is. "
            "Also, give me a single, relevant, and funny hashtag for the image. "
            "Format the result like:\n\n"
            "AGE: 20-23\n"
            "GENDER: MALE\n"
            "MESSAGE: Looks like he just finished yelling at kids to get off his lawn.\n"
            "HASHTAG: #OldManYellsAtCloud"
        )

        # Generate content using the prompt and the image
        response = model.generate_content([prompt, img])
        
        # Return the generated text and a 200 OK status
        return response.text, 200

    except Exception as e:
        # Log any exceptions that occur during Gemini processing for debugging
        print(f"Gemini processing failed: {e}")
        # Return an error message and a 500 Internal Server Error status
        return f"Gemini processing failed: {e}", 500

if __name__ == "__main__":
    # Run the Flask application in debug mode for development
    # This automatically reloads the server on code changes and provides detailed error messages.
    app.run(debug=True)
