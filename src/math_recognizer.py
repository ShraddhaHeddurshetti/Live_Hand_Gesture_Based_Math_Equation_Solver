import cv2
import numpy as np
import easyocr
from sympy import sympify, solve, diff, integrate, Symbol
from sympy.parsing.sympy_parser import parse_expr

class MathRecognizer:
    def __init__(self):
        # Initialize EasyOCR with restricted character set for math
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.x = Symbol('x')  # Default variable for calculus operations
        
    def preprocess_image(self, img):
        """Preprocess image for better OCR results (handwriting friendly)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold is better for uneven handwriting strokes
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to remove noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh
    
    def recognize_math(self, img):
        """Recognize mathematical expression from image"""
        preprocessed = self.preprocess_image(img)

        # Restrict OCR to math characters only
        result = self.reader.readtext(
            preprocessed,
            allowlist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()"
        )
        
        if not result:
            return None
            
        # Combine all detected text
        expression = ' '.join([text[1] for text in result])
        return self._clean_expression(expression)
    
    def solve_expression(self, expression):
        """Solve / evaluate / differentiate / integrate"""
        try:
            expr = expression.lower().replace(" ", "")
            
            # Differentiation
            if expr.startswith("diff(") and expr.endswith(")"):
                inner = expr[5:-1]
                parsed = parse_expr(inner)
                result = diff(parsed, self.x)
                return f"d/dx {inner} = {result}"
            
            # Integration
            if expr.startswith("integrate(") and expr.endswith(")"):
                inner = expr[10:-1]
                parsed = parse_expr(inner)
                result = integrate(parsed, self.x)
                return f"∫ {inner} dx = {result} + C"
            
            # Equation solving
            if "=" in expr:
                left, right = expr.split('=')
                equation = parse_expr(f"{left}-({right})")
                solution = solve(equation)
                return f"x = {solution}"
            
            # Normal arithmetic expression
            result = sympify(expression)
            return f"{expression} = {result}"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _clean_expression(self, expression):
        """Fix common OCR mistakes"""
        replacements = {
            '×': '*',    # Multiplication
            'X': 'x',    # Capital X → variable x
            '𝑥': 'x',    # Unicode italic x → x
            'χ': 'x',    # Greek chi misread → x
            'O': '0',    # Capital O → zero
            'o': '0',    # Small o → zero
            'l': '1',    # Small L → one
            '|': '1',    # Vertical bar → one
            '−': '-',    # Minus
            '—': '-',    # Long dash
            '–': '-',    # En dash
            '[': '(',
            ']': ')',
            '{': '(',
            '}': ')'
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
            
        return expression.strip()
