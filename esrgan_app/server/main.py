import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import RedirectResponse, FileResponse

import cv2
import numpy as np
import tempfile

from esrgan_app.components import generate_SR_image

app_title = "Enhanced Super Resolution GAN - PoC API"
app_desc = """<h2>By: Oscar Suarez - o@oszl.xyz</h2>
<h3>ESRGAN model at the `esrgan/image` endpoint</h3>
<br>Even when some foreseeable exceptions are handled, this is a Proof of Concept, not a properly tested - production ready code.

<br> API developed over the winner model of the PIRM 2018 Challenge from the European Conference on Computer Vision Workshops. https://arxiv.org/abs/1809.00219
"""

app = FastAPI(title=app_title, description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/esrgan/image", response_class=FileResponse)
async def predict_api(file: UploadFile = File(...)):
    # Override to disable the default use of CUDA in PyTorch at esrgan.serve_model.generate_SR_image().
    # (False: will try to set torch.device = CUDA)
    disable_gpu = False

    #Separate file name and extension
    file_name = file.filename.split(".")

    #Check extension
    valid_extension = file_name[-1] in ("jpg", "jpeg", "png")
    if not valid_extension:
        raise HTTPException(status_code=415, detail="Image must be in  jpg or png format.") 
    
    #Generate Super Resolution image
    try:
        prediction = await generate_SR_image(src_image=file, gpu_override=disable_gpu)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Couldn't process the image. Try another one.")

    #Generate result file
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        cv2.imwrite(FOUT.name, prediction)

    #Return the file
    new_file_name = file_name[0]+"_HighRes."+file_name[1]
    return FileResponse(FOUT.name, media_type="image/png", filename=new_file_name)
    

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
