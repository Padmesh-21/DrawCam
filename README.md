# DrawCam - Virtual Finger Drawing with Hand Gesture Recognition

A real-time hand gesture-based drawing application that lets you draw on your screen using just your hand and webcam. Powered by MediaPipe hand detection and OpenCV.

## Overview

DrawCam uses computer vision and hand gesture recognition to create an intuitive drawing experience. Point with your index finger and fold your other fingers to start drawing. Open your hand to stop. It's that simple!

## Features

- 🎨 **Gesture-Based Drawing** - Draw using your index finger when you close your hand (excluding the index finger)
- 🗑️ **Quick Clear** - Clear the canvas by moving your index finger to the top-left corner (within a small zone)
- 🎥 **Real-Time Feed** - Live camera feed with drawing canvas overlay powered by Pygame
- 🟡 **Vibrant Colors** - Draw with yellow color for excellent visibility on the canvas

## Requirements

- Python 3.7+
- Webcam

### Dependencies

- `mediapipe` - Hand detection and tracking
- `opencv-python` - Video capture and processing
- `numpy` - Numerical operations
- `pygame` - Display and rendering

## Installation

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd DrawCam
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python Main.py
   ```

2. Allow access to your webcam when prompted

3. Start drawing using hand gestures (see Gesture Guide below)

4. Close the window to exit

## Gesture Guide

| Action                     | Gesture                                        |
| -------------------------- | ---------------------------------------------- |
| **Start Drawing**          | Index finger up, other fingers folded          |
| **Stop Drawing**           | Open hand or all fingers up                    |
| **Clear Canvas**           | Move index finger to top-left (x < 50, y < 50) |
| **Force Clear (Keyboard)** | Press `C` key                                  |

## Tips for Best Results

- Ensure good lighting for accurate hand detection
- Keep your hand within the camera's view
- Use a contrasting background for better gesture recognition
- Position your webcam at a comfortable angle

## Project Structure

- `Main.py` - Main application script with gesture recognition and drawing logic
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation
