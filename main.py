"""
Main entry point for the CAPTCHA Puzzle Benchmark application.

This module provides a simple entry point that can be used to run the Flask application.
For production use, run 'app.py' directly or use gunicorn.
"""

from app import app

if __name__ == "__main__":
    import os
    
    # For local development
    if os.environ.get("DEVELOPMENT"):
        app.run(debug=True, host="127.0.0.1", port=7860)
    else:
        # For production
        app.run(host="127.0.0.1", port=7860)
