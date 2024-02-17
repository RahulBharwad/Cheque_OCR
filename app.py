import re
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode
import requests
import pytesseract
from datetime import timedelta
from flask_session import Session
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key for flash messages

class ImageProcessingApp:
    def __init__(self):
        self.image_path = None
        self.threshold = 0.75

    def process_image(self):
        if not self.image_path:
            flash("Please select an image or provide a URL.", 'error')
            return None, None

        # Load the source image
        source_path = r"P_80204011000000927.tiff"
        source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            flash(f"Unable to read the source image {source_path}", 'error')
            return None, None

        # Load the target image
        try:
            if self.image_path.startswith(('http://', 'https://')):
                response = requests.get(self.image_path)
                uploaded_image = Image.open(BytesIO(response.content))
                # Convert the image to grayscale using OpenCV
                target_image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2GRAY)
            else:
                target_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            flash(f"Error loading the image: {e}", 'error')
            return None, None

        # Resize the target image to match the dimensions of the source image
        target_image = cv2.resize(target_image, (source_image.shape[1], source_image.shape[0]))

        # Ensure both images have the same depth and type
        target_image = cv2.convertScaleAbs(target_image)

        # Load all template images in the specified folder
        templates_folder = r"C:\Users\admin\Desktop\Sample_model\logo detection\Logos"
        template_files = [f for f in os.listdir(templates_folder) if f.endswith((".jpg", ".jpeg", ".png", ".tiff"))]

        # Set the threshold for template matching
        threshold = self.threshold

        # Variable to check if any template is detected
        template_detected = False

        # Initialize variables to store the best match result and corresponding template name
        best_match_result = -1
        best_template_name = None
        best_locations = None  # Variable to store locations for the template with the highest match result

        # Process each template
        for template_file in template_files:
            template_path = os.path.join(templates_folder, template_file)
            template_name = os.path.splitext(template_file)[0].split("_")[0]

            # Load the template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            template = cv2.convertScaleAbs(template)

            # Apply template matching
            result = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))

            # If any match is found
            if locations:
                template_detected = True

                # Get the best match result and corresponding template name
                best_result_for_template = np.max(result)
                if best_result_for_template > best_match_result:
                    best_match_result = best_result_for_template
                    best_template_name = template_name
                    best_locations = locations

        # If no template is detected, display a message without showing the image
        if not template_detected:
            flash("Previous report:-Logo not detected", 'info')
            return target_image, None

        # Draw rectangles around the detected areas for the template with the best match result
        for loc in best_locations:
            x, y = loc
            h, w = template.shape[:2]  # Use template.shape to get dimensions
            cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("Template detected:", best_template_name)

        return target_image, best_template_name
    

    def read_coordinates_from_json(self, best_template_name):
        json_file_path = "crop_coordinates.json"  
        try:
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                for item in data.get("data", []):
                    if item["template_name"] == best_template_name:
                        return item["coordinates"]
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
        return None


    def crop_and_extract_name(self):
        try:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # Call the process_image method
            target_image, best_template_name = self.process_image()

            # Check if template_name is None
            if best_template_name is None:
                print("Template not detected. Cannot proceed with cropping and extraction.")
                return "Not working for this bank"

            # Retrieve coordinates for the detected template_name from the JSON file
            coordinates_from_json = self.read_coordinates_from_json(best_template_name)

            # Check if coordinates_from_json is None
            if coordinates_from_json is None:
                print("Coordinates not found in JSON for the detected template_name.")
                return "Not working for this bank"

            # Initialize the extracted name
            extracted_name = ""

            # Iterate over each set of coordinates
            for name_coordinates in coordinates_from_json:
                # Convert coordinates_from_json to tuple
                name_coordinates = tuple(map(int, name_coordinates))

                # Cropping coordinates for Name (x1, y1, x2, y2)
                cropped_image_name = Image.open(self.image_path).crop(name_coordinates)

                # Extract text using pytesseract
                extracted_name = pytesseract.image_to_string(cropped_image_name, lang='eng')

                # Further processing to clean extracted name
                if extracted_name:
                    characters_to_replace = ["For", "for", "FOR"]
                    for char in characters_to_replace:
                        extracted_name = extracted_name.replace(char, " ")

                    extracted_name = extracted_name.replace("&", "And")
                    extracted_name = re.sub(r'\b[a-z]+\b', ' ', extracted_name)
                    extracted_name = re.sub(r'[^a-zA-Z ]', ' ', extracted_name)
                    
                    # Check if extracted name is not empty after cleaning
                    if extracted_name.strip():
                        break  # If text is found, break the loop

            if not extracted_name:
                print("Warning: No Name found.")
                return "No Name found"

            print(f"Extracted Name: {extracted_name.strip()}")
            return extracted_name.strip()

        except Exception as e:
            print(f"Error processing {self.image_path} for Name: {str(e)}")
            return "Not working for this bank"

    def set_image_path(self, image_path):
        self.image_path = image_path


        
# Move this instantiation outside the route functions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)  # Adjust as needed
Session(app)

@app.route('/get_details', methods=['GET'])
def get_details():
    processing_app = session.get('processing_app', ImageProcessingApp())

    if processing_app.image_path:
        # Call the function to crop and extract name from the image
        extracted_name = processing_app.crop_and_extract_name()

        # Return the extracted name as JSON
        return jsonify({'extracted_name': extracted_name})

    return jsonify({'error': 'Image path not set'})
    

@app.route('/', methods=['GET', 'POST'])
def index():
    processing_app = session.get('processing_app', ImageProcessingApp())

    if request.method == 'POST':
        uploaded_image = request.files.get('image')
        image_url = request.form.get('image_url', '').strip()

        print("Request Form Data:", request.form)
        print("Uploaded Image:", uploaded_image)
        print("Image URL:", image_url)

        if not (uploaded_image or image_url):
            flash("Please select a file or provide a URL", 'warning')
        else:
            if uploaded_image:
                if uploaded_image.filename == '':
                    flash("No file selected", 'error')
                else:
                    temp_image_folder = os.path.join(os.getcwd(), "temp_image")
                    if not os.path.exists(temp_image_folder):
                        os.makedirs(temp_image_folder)

                    uploaded_image_path = os.path.join(temp_image_folder, "temp_image.jpg")
                    uploaded_image.save(uploaded_image_path)

                    print("Saved uploaded image to:", uploaded_image_path)

                    processing_app.image_path = uploaded_image_path
            elif image_url:
                processing_app.image_path = image_url

            print("Processing image path:", processing_app.image_path)

            processed_image, output_text = processing_app.process_image()
            print("Processed Image:", processed_image)
            print("Output Text:", output_text)

            if processed_image is not None:
                _, buffer = cv2.imencode('.jpg', processed_image)
                img_str = b64encode(buffer).decode('utf-8')
                img_data = f'data:image/jpeg;base64,{img_str}'
                session['processing_app'] = processing_app
                return render_template('result.html', img_data=img_data, output_text=output_text)

            print("Processing failed. Image path:", processing_app.image_path)

    elif request.method == 'GET':
        processing_app.image_path = None
        flash("Restarted. Please select an image or provide a URL.", 'info')

    session['processing_app'] = processing_app
    return render_template('index.html')

if __name__ == '__main__':
    app.run(ssl_context='adhoc', debug=True)
