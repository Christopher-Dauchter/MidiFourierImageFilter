import numpy as np
import cv2
import mido
from threading import Thread
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def apply_fft_filter(img, x_radius, y_radius, angle, center_x,center_y):
    if img is None:
        print("Image data is None.")
        return None, None
    
    # FFT to convert the spatial domain to frequency domain
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    rows, cols = img.shape
    # Use the provided center_x and center_y for the ellipse center
    center = (center_x, center_y)
    axes = (x_radius, y_radius)  # Major and minor axes of the ellipse
    
    # Create an elliptical mask with the updated center
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, thickness=-1)
    
    # Apply mask to the shifted FFT
    fshift_filtered = fshift * mask
    # Inverse FFT to convert back to spatial domain
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    
    # Normalize the filtered image to range [0, 255]
    img_filtered_norm = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to uint8
    img_filtered_uint8 = np.uint8(img_filtered_norm)
    
    # Calculate the magnitude spectrum of the filtered image for visualization
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
    magnitude_spectrum_filtered_norm = cv2.normalize(magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_filtered_uint8, np.uint8(magnitude_spectrum_filtered_norm)

def display_images(original_img, filtered_img, frequency_domain):
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Filtered Image', filtered_img)
    cv2.imshow('Frequency Domain with Filter Applied', frequency_domain)
    cv2.waitKey(1)

def midi_listener(port_name, img):
    x_radius = 50  # Default radius
    y_radius = 50
    angle = 0  # Default angle
    center_x = img.shape[1] // 2  # Default center x based on image width
    center_y = img.shape[0] // 2  # Default center y based on image height
    with mido.open_input(port_name) as inport:
        print("Listening for MIDI messages. Press CTRL+C to exit.")
        for msg in inport:
            if msg.type == 'pitchwheel':
                # Adjust radius and angle based on MIDI channel
                if msg.channel == 0:
                    x_radius = round(map_value(msg.pitch, -8192, 8191, 1, 100))
                elif msg.channel == 1:
                    y_radius = round(map_value(msg.pitch, -8192, 8191, 1, 100))
                elif msg.channel == 2:
                    angle = round(map_value(msg.pitch, -8192, 8191, 0, 360))
                # Adjust ellipse center based on MIDI channels 4 and 5
                elif msg.channel == 3:  # Channel 4 for center X
                    center_x = round(map_value(msg.pitch, -8192, 8191, 0, img.shape[1]))
                elif msg.channel == 4:  # Channel 5 for center Y
                    center_y = round(map_value(msg.pitch, -8192, 8191, 0, img.shape[0]))

                print(f"Ellipse params: radii=({x_radius}, {y_radius}), angle={angle}, center=({center_x}, {center_y})")
                filtered_img, frequency_domain = apply_fft_filter(img, x_radius, y_radius, angle, center_x, center_y)
                if filtered_img is not None and frequency_domain is not None:
                    display_images(img, filtered_img, frequency_domain)

def map_value(v, old_min, old_max, new_min, new_max):
    return ((v - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

def main():
    Tk().withdraw()  # Hide the main tkinter window
    image_path = askopenfilename()  # Open file dialog to select an image
    if not image_path:
        print("No image selected, exiting...")
        return
    
    img = cv2.imread(image_path, 0)  # Read the image
    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    input_names = mido.get_input_names()
    print("Available MIDI input ports:")
    for name in input_names:
        print(name)

    port_name = input("Enter the name of the MIDI input port to use: ")
    midi_listener_thread = Thread(target=midi_listener, args=(port_name, img))
    midi_listener_thread.start()

    try:
        while midi_listener_thread.is_alive():
            time.sleep(1)
    finally:
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
        # Add any additional cleanup actions here
        print("Cleanup completed, exiting...")
if __name__ == "__main__":
    main()
