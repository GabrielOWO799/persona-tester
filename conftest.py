import os
import sys

# 让 `from tools...` 在 pytest 下可导入（把项目根目录加入路径）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
