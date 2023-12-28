import os
import subprocess

def run_main_py():
    '''运行run.py，使得main.py在main.py所在工作夹里运行'''
    # 获取当前脚本所在的文件夹路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建main.py的完整路径
    main_py_path = os.path.join(script_dir, 'code', 'main.py')
    
    # 构建code文件夹的完整路径
    code_dir = os.path.join(script_dir, 'code')
    
    # 使用subprocess模块运行main.py，并将code文件夹作为工作路径
    subprocess.run(['python', main_py_path], cwd=code_dir)

if __name__ == "__main__":
    run_main_py()
