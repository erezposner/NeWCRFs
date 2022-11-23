


from pathlib import Path
model_zoo_path = '../model_zoo/swin_transformer'
Path(model_zoo_path).mkdir(parents=True,exist_ok=True)
import os
os.chdir(model_zoo_path)
print(Path.cwd())
os.system('wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth')