import io
import PIL.Image
from flask import Flask, redirect, request, Response

import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyBJqnwsEDyhXkD73L04O7wxYHIxfCJLETU")

app = Flask(__name__)

@app.route('/')
def index():
    # Redirect to your frontend (Next.js)
    return redirect("http://localhost:3000/tools/genderage")

@app.route('/gemini_upload', methods=['POST'])
def gemini_upload():
    if 'gemini_image' not in request.files:
        return "No Gemini image uploaded", 400

    file = request.files['gemini_image']
    if file.filename == '':
        return "No Gemini image selected", 400

    image_bytes = file.read()

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = PIL.Image.open(io.BytesIO(image_bytes))

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

        response = model.generate_content([prompt, img])
        return response.text, 200

    except Exception as e:
        print(f"Gemini processing failed: {e}")
        return f"Gemini processing failed: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
