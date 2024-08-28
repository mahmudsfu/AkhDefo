'''
### sources

#######

'https://chiefcam.com/resources/images/frame.jpg',
'https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4',
'https://chiefcam.com/video/hls/live/1080p/index.m3u8', 

'''

from flask import Flask, render_template, Response
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import cv2


# Try to import the external function; handle failure gracefully
try:
    from akhdefo_functions import measure_displacement_from_camera
except ImportError as e:
    print(f"Error importing 'akhdefo_functions': {e}")

app = Flask(__name__)

# Record server startup time
startup_time = datetime.utcnow().isoformat()

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

def resilient_measure_displacement_from_camera(*args, **kwargs):
    """Attempts to measure displacement from camera with resilience, incorporating retries."""
    max_attempts = 1000
    attempt = 0
    last_successful_result = None
    while attempt < max_attempts:
        try:
            result = measure_displacement_from_camera(*args, **kwargs)
            yield from result
            last_successful_result = result
            break
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            attempt += 1
            if attempt == max_attempts:
                logger.error("Maximum attempts reached, stopping.")
                if last_successful_result is not None:
                    yield from last_successful_result
                else:
                    yield (None, None)
                break

def generate_frames1(feed_type):
    frames = resilient_measure_displacement_from_camera(
        hls_url='https://chiefcam.com/video/hls/live/1080p/index.m3u8',
        alpha=0.5,
        save_output=True,
        output_filename='example1.mp4',
        ssim_threshold=0.2,
        pyr_scale=0.5,
        levels=100,
        winsize=32,
        iterations=15,
        poly_n=7,
        poly_sigma=1.5,
        flags=1,
        show_video=False,
        streamer_option='mag'
    )

    for frame, g in frames:
        if frame is None and g is None:
            continue
        selected_frame = frame if feed_type == 1 else g
        ret, buffer = cv2.imencode('.jpg', selected_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/startup-time')
def get_startup_time():
    return {'startup_time': startup_time}

@app.route('/video1')
def video1():
    return Response(generate_frames1(feed_type=1), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_frames1(feed_type=2), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Pass the server startup time to the template
    return render_template('index.html', startup_time=startup_time)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
