from flask import Flask, render_template, Response
import cv2
import argparse
import akhdefo_functions

class VideoStreamer:
    """
    A Flask-based web application for streaming video from a camera or video file.

    This class encapsulates the functionalities of streaming video using the OpenCV library
    and custom processing from the 'akhdefo_functions' module. It sets up a Flask server
    with routes to handle video streaming and the main index page.

    Attributes:
        args: Command-line arguments for configuration.
        app: The Flask application instance.

    Methods:
        generate_frames1(): Generates and yields encoded frames with specific processing parameters.
        generate_frames2(): Similar to generate_frames1() but with different parameters.
        setup_routes(): Sets up the Flask routes for the application.
        run(): Starts the Flask server.

    Examples:
        To run the application, first ensure that all dependencies are installed. Then, execute
        the script from the command line with the necessary arguments. For example:

            python video_streamer.py --port 5000

        This command runs the Flask server on port 5000. You can access the main page at
        `http://localhost:5000/`. For streaming video, access `http://localhost:5000/video1`
        or `http://localhost:5000/video2`.

    Note:
        Ensure that the 'akhdefo_functions' module and other dependencies are properly installed
        and accessible to the script. If using custom modules, they should be in the same directory
        as the script or in a location where Python can find them.
        
    """
    def __init__(self):
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Process command line arguments')
        parser.add_argument('--hls_url', default=None, help='HLS URL or 0 for PC webcam or path to a video file')
        # Add other arguments...
        self.args = parser.parse_args()

        # Ensure that the custom module is properly imported
        try:
            from akhdefo_functions import measure_displacement_from_camera
            self.measure_displacement_from_camera = measure_displacement_from_camera
        except ImportError as e:
            print(f"Error importing 'akhdefo_functions': {e}")

        self.app = Flask(__name__)
        self.setup_routes()

    def generate_frames1(self):
        frames = self.measure_displacement_from_camera(
            hls_url=self.args.hls_url,
            # Add other parameters...
        )
        for frame, g in frames:
            # Frame generation logic...
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def generate_frames2(self):
        # Similar to generate_frames1 but with different parameters
        pass

    def setup_routes(self):
        @self.app.route('/video1')
        def video1():
            return Response(self.generate_frames1(), content_type='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/video2')
        def video2():
            return Response(self.generate_frames2(), content_type='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/')
        def index():
            return render_template('index.html')

    def run(self):
        self.app.run(host="0.0.0.0", port=self.args.port, debug=True)

if __name__ == "__main__":
    streamer = VideoStreamer()
    streamer.run()
