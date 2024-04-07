from flask import Flask, render_template, Response
import cv2
import akhdefo_functions
from akhdefo_functions.Akhdefo_Utilities import measure_displacement_from_camera
def run_flask_app():
    """
    Function to run the Flask app and prompt the user for input.

    Parameters:
    - Execute this function in your Python environment.
    - Enter the desired port number for the Flask app when prompted.
    - Press Enter without providing a value to use the default port 80.
    - Access Frame 1 by opening a web browser and visiting the following URL:
    - http://your_server_ip/video1?src_video_url=https://your_video_source_url_frame1
    - Replace 'your_server_ip' with your server's IP address or domain, and 'your_video_source_url_frame1' with the URL of the video source for Frame 1.
    - Access Frame 2 by opening a web browser and visiting the following URL:
    - http://your_server_ip/video2?src_video_url=https://your_video_source_url_frame2 , Replace 'your_server_ip' with your server's IP address or domain, and 'your_video_source_url_frame2' with the URL of the video source for Frame 2.
    
    """
    
   

    try:
        from akhdefo_functions.Akhdefo_Utilities import measure_displacement_from_camera
    except ImportError as e:
        print(f"Error importing 'akhdefo_functions': {e}")

    app = Flask(__name__)

    def generate_frames1():
        hls_url=input("Enter src_video_url: (default 0 for live webcam) ") 
        frames = measure_displacement_from_camera(
                    hls_url=hls_url,  #'https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4', 
                    alpha=0.5,  # Change the alpha value as required
                    save_output=False, 
                    output_filename='example.mp4', 
                    ssim_threshold=0.75, 
                    pyr_scale=0.5, 
                    levels=100, 
                    winsize=120, 
                    iterations=15,
                    poly_n=7, 
                    poly_sigma=1.5, 
                    flags=1, 
                    show_video=False, 
                    streamer_option='mag'
                )

        for frame, g in frames:
            if frame is None or g is None:
                print("Error: Could not retrieve frame or g.")
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            retg, bufferg = cv2.imencode('.jpg', g)
            g = bufferg.tobytes()
            
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            
    ##################
    def generate_frames2():
        hls_url=input("Enter src_video_url: (default 0 for live webcam) ") 
        frames = akhdefo_functions.measure_displacement_from_camera(
                    hls_url=  hls_url, #'https://chiefcam.com/resources/images/frame.jpg',
                    #'https://chiefcam.com/resources/images/frame.jpg',
                    #'https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4',
                    #'https://chiefcam.com/video/hls/live/1080p/index.m3u8', 
                    alpha=0.01,  # Change the alpha value as required
                    save_output=False, 
                    output_filename='example.avi', 
                    ssim_threshold=0.8, 
                    pyr_scale=0.5, 
                    levels=100, 
                    winsize=120, 
                    iterations=15,
                    poly_n=7, 
                    poly_sigma=1.5, 
                    flags=1, 
                    show_video=False, 
                    streamer_option='mag'
                )

        for frame, g in frames:
            if frame is None or g is None:
                print("Error: Could not retrieve frame or g.")
                continue
            retg, bufferg = cv2.imencode('.jpg', g)
            g = bufferg.tobytes()
            
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + g + b'\r\n')

    @app.route('/video1')
    def video1():
        
        f=generate_frames1()
        
        return Response(f, content_type='multipart/x-mixed-replace; boundary=frame')

    

    @app.route('/video2')
    def video2():
        g=generate_frames2()
        
        return Response(g, content_type='multipart/x-mixed-replace; boundary=frame')
    

    @app.route('/')
    def index():
        return render_template('index.html')

   
        
    app.run(host="0.0.0.0", port=800, debug=True)
    
    
    
    
    
    
    
    # from flask import Flask, render_template, Response, request
    # import cv2
    # import akhdefo_functions

    # try:
    #     from akhdefo_functions import measure_displacement_from_camera
    # except ImportError as e:
    #     print(f"Error importing 'akhdefo_functions': {e}")

    # app = Flask(__name__)

    # def generate_frames1(src_video_url):
    #     """
    #     Generate video frames for streaming from the provided video source (Frame 1).

    #     Args:
    #         src_video_url (str): The URL of the video source for Frame 1.

    #     Returns:
    #         generator: A generator that yields video frames as multipart responses for Frame 1.
            
    #     """
    #     frames = measure_displacement_from_camera(
    #         hls_url=src_video_url,
    #         alpha=0.5,
    #         save_output=False,
    #         output_filename='example.mp4',
    #         ssim_threshold=0.75,
    #         pyr_scale=0.5,
    #         levels=100,
    #         winsize=120,
    #         iterations=15,
    #         poly_n=7,
    #         poly_sigma=1.5,
    #         flags=1,
    #         show_video=False,
    #         streamer_option='mag'
    #     )

    #     for frame, g in frames:
    #         if frame is None or g is None:
    #             print("Error: Could not retrieve frame or g.")
    #             continue
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()

    #         yield (b'--frame\r\n'
    #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # def generate_frames2(src_video_url):
    #     """
    #     Generate video frames for streaming from the provided video source (Frame 2).

    #     Args:
    #         src_video_url (str): The URL of the video source for Frame 2.

    #     Returns:
    #         generator: A generator that yields video frames as multipart responses for Frame 2.
            
    #     """
    #     frames = akhdefo_functions.measure_displacement_from_camera(
    #         hls_url=src_video_url,
    #         alpha=0.01,
    #         save_output=False,
    #         output_filename='example.avi',
    #         ssim_threshold=0.8,
    #         pyr_scale=0.5,
    #         levels=100,
    #         winsize=120,
    #         iterations=15,
    #         poly_n=7,
    #         poly_sigma=1.5,
    #         flags=1,
    #         show_video=False,
    #         streamer_option='mag'
    #     )

    #     for frame, g in frames:
    #         if frame is None or g is None:
    #             print("Error: Could not retrieve frame or g.")
    #             continue
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()

    #         yield (b'--frame\r\n'
    #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # @app.route('/video1')
    # def video1(src_video_url):
    #     """
    #     Route for streaming video from the user-provided video source (Frame 1).

    #     Returns:
    #         Response: A Flask Response object for video streaming (Frame 1).
            
    #     """
    #     #src_video_url = request.args.get('src_video_url')
        
    #     f = generate_frames1(src_video_url)
    #     return Response(f, content_type='multipart/x-mixed-replace; boundary=frame')

    # @app.route('/video2')
    # def video2(src_video_url):
    #     """
    #     Route for streaming video from the user-provided video source (Frame 2).

    #     Returns:
    #         Response: A Flask Response object for video streaming (Frame 2).
            
    #     """
    #     #src_video_url = request.args.get('src_video_url')
    #     g = generate_frames2(src_video_url)
    #     return Response(g, content_type='multipart/x-mixed-replace; boundary=frame')

    # @app.route('/')
    # def index():
    #     """
    #     Route for rendering the index.html template.

    #     Returns:
    #         str: The rendered HTML template.
            
    #     """
    #     return render_template('index.html')

    
    # src_video_url=input("Enter src_video_url: (default 0 for live webcam) ")  
    # port = input("Enter the port number (default is 80): ")
    
    # video1(src_video_url)
    # video2(src_video_url)
    
   

    # app.run(host="0.0.0.0", port=port, debug=False)

   
