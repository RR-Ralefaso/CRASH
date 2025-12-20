import cv2
import time
from Crash import CrashDetector
from config import CAMERA_ID, WINDOW_NAME
from outputting import reset_output_file, write_detection_output  # Import the file functions

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("Camera error! Try CAMERA_ID = 0, 1, 2 in config.py")
        return
    
    # Reset the output file (delete and create a fresh one)
    reset_output_file()

    # Get the maximum resolution supported by the camera
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Default width (e.g., 640, 1280)
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Default height (e.g., 480, 720)
    
    # Adjustable resolution: set to max supported by the camera or custom
    width = max_width  # Max resolution width
    height = max_height # Max resolution height
    
    # Set the camera resolution to the desired width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Faster shutter
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)      # Brighter image
    cap.set(cv2.CAP_PROP_CONTRAST, 120)        # Sharper image
    cap.set(cv2.CAP_PROP_SHARPNESS, 200)       # Crisp edges
    
    # Get actual properties after setting (to verify resolution)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {width}x{height} @ {fps:.1f}FPS")
    
    detector = CrashDetector()
    
    # Open the file to log the start time (this can be done in outputting.py, too)
    write_detection_output("Starting detection session.")
    
    print("CRASH DETECTOR - Crystal Clear! 'q' to quit, 'd' toggle detection")
    detect_on = True
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # RAW CAMERA (NO RESIZE) = CLEAR IMAGE
        display_frame = frame.copy()
        
        if detect_on:
            # Resize ONLY for YOLO (not display)
            small_frame = cv2.resize(frame, (640, 480))
            detections = detector.detect(small_frame)  # Assuming detect returns a list of detections
            
            # Scale back up for display
            display_frame = cv2.resize(detections['annotated_frame'], (width, height))
            
            # Assuming `detections['objects']` contains the list of detected objects and their details
            if detections['objects']:
                for obj in detections['objects']:
                    object_type = obj['type']  # e.g., "car", "bicycle", etc.
                    coordinates = obj['bbox']  # e.g., [x1, y1, x2, y2] for the bounding box
                    detection_info = f"Detected {object_type} at {coordinates}."
                    
                    # Log what the detector saw in the file
                    write_detection_output(detection_info)
            else:
                write_detection_output("No objects detected.")
        else:
            # Pure raw camera feed
            cv2.putText(display_frame, "RAW CAMERA (Press 'd' for detection)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS overlay
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            start_time = time.time()
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            detect_on = not detect_on
            print("Detection:", "ON" if detect_on else "OFF")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
