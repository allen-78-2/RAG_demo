from modelscope import snapshot_download
import os


# 模型下载
model_dir = snapshot_download('BAAI/bge-large-zh-v1.5', local_dir=os.path.join(os.getcwd(), 'BAAI', 'bge-large-zh-v1.5'))  # os.getcwd() 当前路径
