# Yolov8 Inference using Streamlit
A web interface for real-time yolo inference using streamlit. It supports CPU and GPU inference, supports both images and videos and uploading your own custom models.

## Features
- **Caches** the model for faster inference on both CPU and GPU.
- Supports uploading model files (<200MB) and downloading models from URL (any size)
- Supports both images and videos.
- Supports both CPU and GPU inference.
- Supports:
  - Custom Classes
  - Changing Confidence
  - Changing input/frame size for videos


## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add the model file to `sample_models/` and change `cfg_model_path` to its path.
```bash
git clone https://github.com/phucnamjose/Python-Streamlit.git
cd Python-Streamlit
source setup.sh
streamlit run app.py --server.port [PORT]
```

## References
https://github.com/moaaztaha/Yolo-Interface-using-Streamlit
