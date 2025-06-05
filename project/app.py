from fastapi import FastAPI
from fastapi.responses import JSONResponse
import subprocess
import os
from fastapi.responses import PlainTextResponse

app = FastAPI()

@app.get("/get_alphabet")
def get_alphabet():
    try:
        result = subprocess.run(['python', 'isl_detection.py'], capture_output=True, text=True, cwd=os.getcwd())
        return JSONResponse(content={"alphabet": result.stdout.strip()}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.get("/get_word")
# def get_word():
#     try:
#         result = subprocess.run(['python', 'deploy-code.py'], capture_output=True, text=True, cwd=os.getcwd())
#         return JSONResponse(content={"word": result.stdout.strip()}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/get_word", response_class=PlainTextResponse)
def get_word():
    try:
        result = subprocess.run(
            ['python', 'deploy-code.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Combine stdout and stderr just in case
        output = result.stdout + result.stderr

        if not output.strip():
            return PlainTextResponse("No output from deploy_code.py", status_code=200)

        return PlainTextResponse(output.strip(), status_code=200)

    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)


@app.get("/test")
def test():
    return {"message": "Hello"}
