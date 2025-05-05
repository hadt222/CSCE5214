from flask import Flask, render_template, send_from_directory, jsonify
import pandas as pd

app = Flask(__name__)

# Load and preprocess log file into dictionary
def load_log_file(log_filename):
    data = []
    with open(log_filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame = int(parts[0].split(':')[1].strip())
            tracker_id = int(parts[1].split(':')[1].strip())
            speed_str = parts[2].split(':')[1].strip().split()[0]
            speed = int(speed_str)
            data.append({'frame': frame, 'tracker_id': tracker_id, 'speed': speed})
    
    df = pd.DataFrame(data)
    grouped = df.groupby('frame')
    log_dict = {}
    for frame, group in grouped:
        log_dict[frame] = group[['tracker_id', 'speed']].to_dict(orient='records')
    return log_dict

# Load both log files
log_data1 = load_log_file('speed_estimation.txt')
log_data2 = load_log_file('speed_estimation2.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video1')
def video1():
    return send_from_directory('.', 'video.mp4')

@app.route('/video2')
def video2():
    return send_from_directory('.', 'video2.mp4')

@app.route('/log_data1')
def get_log_data1():
    return jsonify(log_data1)

@app.route('/log_data2')
def get_log_data2():
    return jsonify(log_data2)

if __name__ == '__main__':
    app.run(debug=True)
