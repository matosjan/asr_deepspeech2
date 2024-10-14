import os
import shutil

import gdown

def download():
    gdown.download("https://drive.google.com/uc?id=1X0wBw9P3uS1ID1eLXPn2AK9t_h1DmQT3")
    os.makedirs("./src/model_weights", exist_ok=True)
    shutil.move("model_best.pth", "./src/model_weights/model_best.pth")


if __name__ == "__main__":
    download()