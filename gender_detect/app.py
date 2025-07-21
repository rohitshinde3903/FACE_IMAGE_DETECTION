import PIL
from flask import Flask, render_template, Response, request


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

import google.generativeai as genai
genai.configure(api_key="AIzaSyBJqnwsEDyhXkD73L04O7wxYHIxfCJLETU")
import PIL.Image
import io

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
        result = response.text

        # Return the raw text directly, not render a template
        return result, 200 # Return text with a 200 OK status

    except Exception as e:
        # Log the error for debugging
        print(f"Gemini processing failed: {e}")
        return f"Gemini processing failed: {e}", 500

if __name__ == "__main__":
    app.run(debug=True) # Run in debug mode for easier development



if __name__ == "__main__":
    app.run()
