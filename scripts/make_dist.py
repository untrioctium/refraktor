import os
import shutil
from glob import glob

def simple_copy(path):
    return {
        "type": "simple",
        "path": path
    }

def file_copy(from_path, to_path):
    return {
        "type": "file",
        "from": from_path,
        "to": to_path
    }

common_dist = [
    simple_copy("assets/denoise_weights/"),
    simple_copy("assets/fonts/"),
    simple_copy("assets/kernels/"),
    simple_copy("assets/static/"),
    simple_copy("assets/templates/"),
    simple_copy("config/"),
    simple_copy("licenses/"),
    simple_copy("bin/")
]

def to_dist(path):
    return os.path.join(os.getcwd(), "dist/", path)

windows_dist = [
    simple_copy("*.dll"),
    file_copy("build/src/refrakt-gui/Release/refrakt-gui.exe", "refrakt-gui.exe")
]

dist = common_dist + windows_dist

for dep in dist:
    if dep["type"] == "simple":
        if "*" in dep["path"]:
            print(f"Copying {dep['path']} to {to_dist(dep['path'])}")
            full_path = os.path.join(os.getcwd(), dep["path"])
            orig_path = os.path.dirname(dep["path"])
            for file in glob(full_path):
                real_file = to_dist(os.path.basename(file))
                print(f"  Copying {file} to {real_file}")
                shutil.copy(file, real_file)

        else:
            print(f"Copying {dep['path']} to {to_dist(dep['path'])}")
            if os.path.isdir(dep["path"]):
                shutil.copytree(dep["path"], to_dist(dep["path"]), dirs_exist_ok=True)
            else:
                shutil.copy(dep["path"], to_dist(dep["path"]))
        
    elif dep["type"] == "file":
        print(f"Copying {dep['from']} to {to_dist(dep['to'])}")
        shutil.copy(dep["from"], to_dist(dep["to"]))
    else:
        raise Exception(f"Unknown dependency type: {dep['type']}")