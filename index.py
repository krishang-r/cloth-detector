from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Define the folder where your React app is located
react_folder = 'moodit-master'

# Define the directory to serve static files from
directory = os.path.join(os.getcwd(), react_folder, 'build')

# Define the route for serving the index.html file
@app.route('/')
def index():
    return send_from_directory(directory, 'index.html')

# Define the route for serving static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(directory, 'static'), filename)

# Define the route for serving the manifest.json file
@app.route('/manifest.json')
def manifest():
    return send_from_directory(directory, 'manifest.json')

# Define the route for serving JavaScript files
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(directory, 'static/js'), filename)

# Define a catch-all route for serving any other static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(directory, filename)

@app.route('/upload')
def upload():
    if 'image' not in request.files:
        return "No image provided", 400

    image = request.files['image']
    image.save(os.path.join(os.getcwd(), 'uploads', image.filename))
    # image.save('uploads/' + image.filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
