import pyautogui
from PIL import ImageGrab
import time

def get_screen_color():
    while True:
        # Get the current mouse cursor's x and y positions
        x, y = pyautogui.position()
        
        # Grab the pixel color at the cursor's current position
        pixel_color = ImageGrab.grab().getpixel((x, y))
        
        # Print the RGB values
        print(f"RGB at position ({x}, {y}): {pixel_color}", end='\r')
        
        # Delay to reduce the frequency of updates
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        get_screen_color()
    except KeyboardInterrupt:
        print("Program terminated by user.")
