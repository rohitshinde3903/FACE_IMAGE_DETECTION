import PIL
from flask import Flask, render_template, Response, request
import numpy as np
from PIL import Image


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
    "Gender must be either MALE or FEMALE. "
    "Write the MESSAGE in Hinglish (Hindi in Roman script) — make it funny, blunt, and savage, as if you're roasting the person. "
    "Add one word rogue-style, hilarious hashtag in **Hindi only**, based on the person’s mood, expression, or vibe. "
    "You can use examples like #Rotalu (sad), #aagkagola (angry), #bhaaicoderhai (neutral), #yellowteethsmile (smile), "
    "but do **not** overuse them — create your own Hindi hashtags matching the facial expression. "
    "Be as blunt and witty as Grok — no sugarcoating, make it roast-worthy. "
    "The hashtag should sound like a viral desi meme. "
    "Format the result like:\n\n"
    "AGE: 20-23\n"
    "GENDER: MALE\n"
    "MESSAGE: Lagta hai ye banda gym sirf selfie lene jaata hai.\n"
    "Your Hashtag: #gymselfiepaglu"
)




        response = model.generate_content([prompt, img])
        result = response.text

        return render_template('gemini_result.html', gemini_prediction=result)

    except Exception as e:
        return f"Gemini processing failed: {e}", 500



if __name__ == "__main__":
    app.run()
