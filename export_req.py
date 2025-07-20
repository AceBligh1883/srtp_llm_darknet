import subprocess
import sys
import os

def export_requirements(file_path="requirements.txt"):
    """
    使用 'pip freeze' 命令导出当前 Python 环境的依赖到指定文件。

    :param file_path: 导出的文件名，默认为 'requirements.txt'
    """
    print("正在导出依赖...")
    try:
        # 确保我们使用的是当前脚本所在环境的 pip
        # sys.executable 指向当前的 python 解释器
        # 使用 [sys.executable, "-m", "pip", "freeze"] 是最稳妥的方式
        command = [sys.executable, "-m", "pip", "freeze"]
        
        # 执行命令并捕获输出
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            encoding='utf-8' # 明确编码以避免乱码
        )
        
        # 获取命令的输出内容
        requirements_content = result.stdout
        
        # 将内容写入文件
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(requirements_content)
            
        full_path = os.path.abspath(file_path)
        print(f"✅ 成功！依赖已导出到: {full_path}")
        print("\n--- 文件内容预览 ---")
        print(requirements_content.strip().split('\n')[0])
        print("...")
        
    except FileNotFoundError:
        print("❌ 错误：找不到 'pip'。请确保 pip 已安装并位于您的 PATH 中。")
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行 'pip freeze' 时出错：\n{e.stderr}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")

if __name__ == "__main__":
    export_requirements()

