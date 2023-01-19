# Enhanced Super Resolution GAN - Proof of Concept API 

## __Technologies used__
![Used](https://skillicons.dev/icons?i=py,fastapi,pytorch)

## __Description__
API developed to serve an Enhanced Super Resolution Generative Adversarial Network model. This (yet to come) [video](https://youtube.com/) contains an explanation of the ideas behind Super Resolution Generative networks, the enhancements of this model and finally, an overview of how to set up and test this API.

## __Quick Test__
#### __Dependencies__
- Python 3
- FastAPI (tested on 0.85.0)
- PyTorch (tested on 1.13.1 [Optional: + CUDA 11.6])
- OpenCV (tested on 4.7.0)
- NumPy

#### __Test the API__
1. Clone the repo.
    ```
    git clone https://github.com/
    ```
2. Configure the  **disable_gpu** variable in `esrgan_app\server\main.py:31` . True = PyTorch will use the CPU. False = PyTorch will try to use CUDA.
3. Run the FastAPI app from Uvicorn. 
    ```
    cd superResolution
    python -m uvicorn esrgan_app.server.main:app
    ```
5. Go to  `http://127.0.0.1:8000` in a browser to get to the **Docs** page.
5. Expand the `esrgan/image` endpoint and click on the **Try it out** button.
6. Upload the png/jpg file and click **Execute**. Some test images are provided under the `test_img` folder.
7. Server answer will load below (download link for the procesed image or error).
8. **CTRL+C** will stop the server.

