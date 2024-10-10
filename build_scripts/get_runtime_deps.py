import os
import tempfile
import shutil
import urllib.request
import zipfile

def simple_dependency(url, path):
    return {
        "type": "simple",
        "url": url,
        "path": path
    }

def zip_dependency(url, *members):
    return {
        "type": "zip",
        "url": url,
        "members": members
    }

def local_dependency(from_path, to_path):
    return {
        "type": "local",
        "from": from_path,
        "to": to_path
    }

common_dependencies = [
    # font files
    simple_dependency("https://github.com/google/material-design-icons/raw/master/font/MaterialIcons-Regular.ttf", "assets/fonts/"),
    simple_dependency("https://github.com/google/material-design-icons/raw/master/font/MaterialIconsOutlined-Regular.otf", "assets/fonts/"),
    simple_dependency("https://github.com/google/material-design-icons/raw/master/font/MaterialIconsSharp-Regular.otf", "assets/fonts/"),
    simple_dependency("https://github.com/google/material-design-icons/raw/master/font/MaterialIconsTwoTone-Regular.otf", "assets/fonts/"),
]

windows_dependencies = [
    # ffmpeg
    zip_dependency(
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        ["ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe", "bin/"],
        ["ffmpeg-master-latest-win64-gpl/LICENSE.txt", "licenses/FFMPEG-LICENSE.txt"],

    ),

    # nvrtc
    zip_dependency(
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.4.99-archive.zip", 
        ["cuda_nvrtc-windows-x86_64-12.4.99-archive/bin/nvrtc64_120_0.dll", "./"],
        ["cuda_nvrtc-windows-x86_64-12.4.99-archive/bin/nvrtc-builtins64_124.dll", "./"],
        ["cuda_nvrtc-windows-x86_64-12.4.99-archive/LICENSE", "licenses/NVRTC-LICENSE.txt"]
    ),

    # nvjpeg
    zip_dependency(
        "https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/windows-x86_64/libnvjpeg-windows-x86_64-12.3.1.89-archive.zip",
        ["libnvjpeg-windows-x86_64-12.3.1.89-archive/bin/nvjpeg64_12.dll", "./"],
        ["libnvjpeg-windows-x86_64-12.3.1.89-archive/LICENSE", "licenses/NVJPEG-LICENSE.txt"]
    ),

    # open image denoise
    local_dependency("build/_deps/oidn-src/bin/OpenImageDenoise.dll", "./"),
    local_dependency("build/_deps/oidn-src/bin/OpenImageDenoise_core.dll", "./"),
    local_dependency("build/_deps/oidn-src/bin/OpenImageDenoise_device_cuda.dll", "./"),
    local_dependency("build/_deps/oidn-src/bin/OpenImageDenoise_device_hip.dll", "./"),
]

dependencies = common_dependencies + windows_dependencies

def path_is_dir(path):
    return path.endswith("/")

def ensure_path_exists(path):
    if path_is_dir(path):
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def filename(path):
    return path.split("/")[-1]

def download_file(url, path):
    ensure_path_exists(path)

    print(f"Downloading {url} to {path}")
    if path_is_dir(path):
        path = os.path.join(path, filename(url))

    urllib.request.urlretrieve(url, path)

print(f'Working directory: {os.getcwd()}')

with tempfile.TemporaryDirectory() as temp_dir:
    for dep in dependencies:
        if dep["type"] == "simple":
            download_file(dep["url"], dep["path"])
        elif dep["type"] == "local":
            print(f"Copying {dep['from']} to {dep['to']}")
            shutil.copy(dep["from"], dep["to"])

        elif dep["type"] == "zip":
            zip_file = os.path.join(temp_dir, filename(dep["url"]))
            download_file(dep["url"], zip_file)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for member in dep["members"]:
                    print(f"  Extracting {member[0]} to {member[1]}")
                    zip_info = zip_ref.getinfo(member[0])
                    target_filename = os.path.basename(zip_info.filename) if path_is_dir(member[1]) else os.path.basename(member[1])
                    target_path = os.path.join(os.getcwd(), os.path.dirname(member[1]))
                    ensure_path_exists(target_path)

                    zip_info.filename = target_filename
                    zip_ref.extract(zip_info, target_path)
        else:
            raise Exception(f"Unknown dependency type: {dep['type']}")
    