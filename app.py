from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from detect import predict

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.txt'):
            return jsonify({'error': 'Only .txt files allowed'}), 400
        
        text = file.read().decode('utf-8')
        results = predict(text)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 500
        
        return jsonify({
            'status': 'success',
            'filename': secure_filename(file.filename),
            'results': {
                'average_similarity': results['average_similarity'],
                'top_matches': results['top_matches']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)