from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_alphabet")
def get_alphabet():
    try:
        result = subprocess.run(
            ['python', 'isl_detection.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return JSONResponse(
            content={"alphabet": result.stdout.strip()},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# @app.get("/get_word", response_class=PlainTextResponse)
# def get_word():
#     try:
#         result = subprocess.run(
#             ['python', 'deploy-code.py'],
#             capture_output=True,
#             text=True,
#             cwd=os.getcwd()
#         )
#         print(result.stdout.strip())
#         return result.stdout.strip()
#     except Exception as e:
#         return f"Error: {str(e)}"
    

@app.get("/get_word", response_class=PlainTextResponse)
def get_word():
    try:
        result = subprocess.run(
            ['python', 'deploy-code.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        output_lines = result.stdout.strip().splitlines()

        # Look for the line containing "video is"
        predicted_line = next((line for line in output_lines if "video is" in line), None)

        if predicted_line:
            # Extract just the predicted word
            predicted_word = predicted_line.split("video is")[-1].strip()
            return predicted_word

        return "Prediction not found"

    except Exception as e:
        return f"Error: {str(e)}"










