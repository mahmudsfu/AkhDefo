from flask import Flask, render_template, Response
import cv2
import akhdefo_functions
import matplotlib
try:
    from akhdefo_functions import measure_displacement_from_camera
except ImportError as e:
    print(f"Error importing 'akhdefo_functions': {e}")

app = Flask(__name__)

def generate_frames1():
    
    frames = measure_displacement_from_camera(
                hls_url= 'https://chiefcam.com/video/hls/live/1080p/index.m3u8', #"https://chiefcam.com/video/hls/live/1080p/index.m3u8", #'https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4', 
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
    
    frames = akhdefo_functions.measure_displacement_from_camera(
                hls_url= "https://chiefcam.com/video/hls/live/1080p/index.m3u8" ,  #'https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4',
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

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=80, debug=True)
    
    matplotlib.pyplot.close()

