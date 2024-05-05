import re
import subprocess
import os

from tqdm import tqdm

import glob
import argparse

import torch

from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.columns import Columns
console = Console()

current_path = os.getcwd()
is_debug = False

Guochenxv888!

def is_package_installed(package_name):
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    packages = result.stdout
    return package_name.lower() in packages.lower()

def get_content(package):
    """Extract text from user dict."""
    status = package["status"]
    name  = package['name']
    return f"[b]{name}[/b]\n{status}"

def check_installed_package():
    console.print(":robot: check packages ", style="white on blue")
    with open(f"{current_path}/requirements.txt","r") as f:
        package_list  = []
        for line in tqdm(f):
            clean_line = line.strip()
            if clean_line:
                if clean_line and not re.match(r'^#', clean_line):
                    if '==' in clean_line:
                        package_name = clean_line.split('==')[0]
                        if not is_package_installed(package_name):
                            # print(f"plz install {package_name}")
                            package_list.append({
                                "name":package_name,
                                "status":"[red]not installed[/]"
                            })
                        else:
                            # print(f"{package_name} installed")
                            package_list.append({
                                "name":package_name,
                                "status":"[green]installed[/]"
                            })
                    else:
                        package_name = clean_line
                        if not is_package_installed(package_name):
                            package_list.append({
                                "name":package_name,
                                "status":"[red]not installed[/]"
                            })
                        else:
                            package_list.append({
                                "name":package_name,
                                "status":"[green]installed[/]"
                            })
    package_renderables = [Panel(get_content(package), expand=True) for package in package_list]
    console.print(Columns(package_renderables))

def check_os():
    console.print(":robot: check OS platform ", style="white on blue")
    os_name = os.name
    if os_name == 'nt':
        print(":robot: OS platform: Windows")
    elif os_name == 'posix':
        print(":robot: OS platform: [blue]Unix[\] (Linux, macOS, etc.)")
    else:
        print(f":robot: OS platform : Other{os_name}")

def check_gpu():
    console.print(":robot: check gpu ", style="white on blue")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    # 如果 CUDA 可用，输出更多的 GPU 信息
    if cuda_available:
        # 获取 GPU 的数量
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs Available: {num_gpus}")

        # 逐个输出每个 GPU 的信息
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def dataset_check():
    pass

# 
def check_module_installed_package():

    console.print(":robot: check modules ", style="white on blue")
    folders_in_cpp = glob.glob(f"{current_path}/openpoints/cpp/*")

    filtered_folders = [folder for folder in folders_in_cpp if not re.search(r'(__pycache__|\.py$)', folder)]
    module_list = []
    for cpp_module_folder in filtered_folders:
        module_name = cpp_module_folder.split("/")[-1]
        # print(glob.glob(f"{cpp_module_folder}/*"))
        if "build" in [ folder.split("/")[-1] for folder in glob.glob(f"{cpp_module_folder}/*")]:
            module_list.append({
                    "name":module_name,
                    "status":"[green]installed[/]"
                })
        else:
             module_list.append({
                    "name":module_name,
                    "status":"[red]not installed[/]"
                })
             
    module_renderables = [Panel(get_content(module), expand=True) for module in module_list]
    console.print(Columns(module_renderables))

def main():
  

    # print(f"The base folder name is: {args.base_folder_name}")

    check_installed_package()
    check_module_installed_package()

    # 判断操作系统
    check_os()

    # gpu 检查
    check_gpu()
    # checkpoint

    # dataset 检查
    # dataset_check()

    # dataset bag


if __name__ == "__main__":
    main()