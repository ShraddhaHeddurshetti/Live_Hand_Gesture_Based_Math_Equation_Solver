import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from hand_detector import HandDetector
from math_recognizer import MathRecognizer
import threading
import queue
import time

class MathSolverApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Live Hand Gesture Based Math Equation Solver Using AI")
        
        # Initialize components
        self.hand_detector = HandDetector()
        self.math_recognizer = MathRecognizer()
        
        # Current state
        self.current_mode = "WRITE"
        self.current_equation = ""
        self.drawing = False
        self.last_point = None
        self.is_writing = False
        self.is_hovering = False  
        
        # Frame queue for threading
        self.frame_queue = queue.Queue(maxsize=3)
        
        # Set fixed dimensions
        self.frame_width = 640
        self.frame_height = 480
        
        # Drawing canvas
        self.canvas_img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Gesture debouncing
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3  
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_video)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start video loop
        self.update_video()
    
    # ------------------ Project Info ------------------
    def show_project_info(self):
        info = """\
DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE
KLE College of Engineering and Technology, Chikodi

PROJECT TITLE:
LIVE HAND GESTURE BASED MATH EQUATION SOLVER USING AI

GROUP MEMBERS:
1. Mr. Darshan K
2. Ms. Shraddha H
3. Ms. Vaishnavi P
4. Ms. Sneha M

GUIDE:
Dr. Bahubali Akiwate
"""
        info_window = tk.Toplevel(self.window)
        info_window.title("Project Info")
        tk.Label(info_window, text=info, justify="left", font=("Arial", 11)).pack(padx=20, pady=20)
    # --------------------------------------------------

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Project Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="LIVE HAND GESTURE BASED MATH EQUATION SOLVER USING AI",
            font=("Arial", 14, "bold"),
            foreground="darkblue"
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=1, column=0, padx=5, pady=5)
        
        # Right sidebar
        self.sidebar = ttk.Frame(self.main_frame)
        self.sidebar.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Mode display
        self.mode_label = ttk.Label(self.sidebar, text="Current Mode: WRITE")
        self.mode_label.pack(pady=5)
        
        # Writing status
        self.writing_label = ttk.Label(self.sidebar, text="Writing: Stopped")
        self.writing_label.pack(pady=5)
        
        # Results display (big + bold)
        result_frame = tk.Frame(self.sidebar, bg="#fdf5e6", bd=2, relief="groove")
        result_frame.pack(pady=10, fill="both", expand=True)
        tk.Label(result_frame, text="Results:", font=("Arial", 12, "bold"), bg="#fdf5e6").pack(anchor="w")
        self.result_text = tk.Text(result_frame, height=8, width=40, wrap="word", font=("Consolas", 11, "bold"), bg="#fdf5e6", fg="black")
        self.result_text.pack(fill="both", expand=True)

        # Clear button
        self.clear_btn = ttk.Button(self.sidebar, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)

        # Project Info button
        self.info_btn = ttk.Button(self.sidebar, text="Project Info", command=self.show_project_info)
        self.info_btn.pack(pady=10)

        # Operation Mode
        self.operation_var = tk.StringVar(value="Solve")
        operation_label = ttk.Label(self.sidebar, text="Select Operation:")
        operation_label.pack(pady=5)
        self.operation_menu = ttk.Combobox(
            self.sidebar, textvariable=self.operation_var, 
            values=["Solve", "Differentiate", "Integrate"], state="readonly"
        )
        self.operation_menu.pack(pady=5)
        
        # Instructions
        instructions = """Gestures:
- ☝️ Index finger up: Write
- ✌️ Two fingers up: Erase
- ✋ All fingers up: Clear
- ✊ Fist: Process (Solve/Differentiate/Integrate)
- 🖖 Three fingers up: Hover (move without drawing)"""
        self.instructions = ttk.Label(self.sidebar, text=instructions, font=("Arial", 9, "underline"))
        self.instructions.pack(pady=20)
        
        # Footer
        project_details = """Dept. of AI & DS | KLE College of Engg & Tech, Chikodi
Guide: Dr. Bahubali Akiwate
Team: Darshan K, Shraddha H, Vaishnavi P, Sneha M"""
        self.footer = ttk.Label(self.main_frame, text=project_details, font=("Arial", 9), foreground="blue")
        self.footer.grid(row=2, column=0, columnspan=2, pady=10)

    # --------------------------------------------------

    def clear_canvas(self):
        self.canvas_img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.last_point = None

    def process_hand_landmarks(self, frame, landmarks):
        """Process hand landmarks for drawing"""
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        if not landmarks:
            self.last_point = None
            return frame
            
        index_finger = landmarks[8]
        x = int(index_finger.x * self.frame_width)
        y = int(index_finger.y * self.frame_height)
        current_point = (x, y)
        
        output_frame = frame.copy()
        
        # Draw fingertip indicators
        if self.current_mode == "WRITE" and self.is_writing:
            cv2.circle(output_frame, current_point, 5, (255, 0, 0), -1)
        elif self.current_mode == "ERASE":
            cv2.circle(output_frame, current_point, 24, (0, 0, 255), 1)
            
        # Handle writing
        if self.current_mode == "WRITE" and self.is_writing:
            if self.last_point is not None:
                cv2.line(self.canvas_img, self.last_point, current_point, (255, 255, 255), 3)
            self.last_point = current_point
        elif self.current_mode == "ERASE":
            cv2.circle(self.canvas_img, current_point, 24, (0, 0, 0), -1)
            self.last_point = None
        else:
            self.last_point = None
        
        # Overlay canvas on video
        try:
            canvas_resized = cv2.resize(self.canvas_img, (self.frame_width, self.frame_height))
            if output_frame.shape == canvas_resized.shape:
                output_frame = cv2.addWeighted(output_frame, 1, canvas_resized, 0.5, 0)
        except cv2.error as e:
            print(f"Error combining frames: {e}")
        
        return output_frame

    def capture_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)

    def update_video(self):
        try:
            current_time = time.time()
            frame = self.frame_queue.get_nowait()
            if frame is None:
                raise queue.Empty
                
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame = cv2.flip(frame, 1)
            
            # Hand detection
            frame = self.hand_detector.find_hands(frame)
            gesture = self.hand_detector.get_gesture()
            
            # Process landmarks
            if self.hand_detector.results.multi_hand_landmarks:
                frame = self.process_hand_landmarks(
                    frame, 
                    self.hand_detector.results.multi_hand_landmarks[0].landmark
                )
            
            # Gesture handling with debounce
            if gesture and (current_time - self.last_gesture_time) > self.gesture_cooldown:
                self.last_gesture_time = current_time
                
                if gesture == "CLEAR":
                    self.clear_canvas()
                elif gesture == "SOLVE":
                    self._process_solve()
                elif gesture == "WRITE":
                    self.is_writing = True
                    self.writing_label.config(text="Writing: Started", foreground="green")
                else:
                    self.is_writing = False
                    self.writing_label.config(text="Writing: Stopped", foreground="red")
                
                if gesture not in ["HOVER"]:
                    self.current_mode = gesture
                    self.mode_label.config(text=f"Current Mode: {self.current_mode}")
            
            # Status overlay
            cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert frame to Tkinter Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = PIL.Image.fromarray(frame_rgb)
            frame_tk = PIL.ImageTk.PhotoImage(image=frame_pil)
            
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in video update: {e}")
        
        self.window.after(33, self.update_video)

    def __del__(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()

    def _process_solve(self):
        """Solve/Differentiate/Integrate and show ONLY latest result"""
        try:
            ocr_img = cv2.bitwise_not(self.canvas_img)
            equation = self.math_recognizer.recognize_math(ocr_img)
            if equation:
                self.current_equation = equation
                operation = self.operation_var.get()

                if operation == "Differentiate":
                    solution = self.math_recognizer.solve_expression(f"diff({equation})")
                elif operation == "Integrate":
                    solution = self.math_recognizer.solve_expression(f"integrate({equation})")
                else:  # Solve
                    solution = self.math_recognizer.solve_expression(equation)

                # 🔥 Clear old results and show only latest
                self.result_text.delete("1.0", "end")
                self.result_text.insert(
                    "end",
                    f"Operation: {operation}\nExpression: {equation}\nResult: {solution}\n"
                )
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: Could not process\n{str(e)}")

def main():
    root = tk.Tk()
    app = MathSolverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
