import json
import numpy as np
import os

def read_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines[3:]:
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def read_images_txt(path):
    images = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines[4::2]:
            parts = line.strip().split()
            image_id = int(parts[0])
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            images[image_id] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": name
            }
    return images

def create_transforms_json(cameras, images):
    frames = []
    for image_id, image in images.items():
        camera = cameras[image["camera_id"]]
        frame = {
            "file_path": image["name"],
            "rotation": [image["qw"], image["qx"], image["qy"], image["qz"]],
            "translation": [image["tx"], image["ty"], image["tz"]],
            "camera_angle_x": 2 * np.arctan(camera["params"][0] / (2 * camera["width"])),
            "camera_angle_y": 2 * np.arctan(camera["params"][0] / (2 * camera["height"]))
        }
        frames.append(frame)
    transforms = {"frames": frames}
    return transforms

def main():
    cameras_path = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/sparse/0/cameras.txt'
    images_path = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/sparse/0/images.txt'

    # Verify the existence of the files before reading
    if not os.path.exists(cameras_path):
        print(f"Error: {cameras_path} does not exist.")
        return
    if not os.path.exists(images_path):
        print(f"Error: {images_path} does not exist.")
        return

    # Print the contents of the directory to verify the files are there
    print("Listing directory content:")
    print(os.listdir('/vol/bitbucket/kt1923/4DGaussians/data/multipleview/office_0/colmap/sparse/0/'))

    print(f"Reading cameras from {cameras_path}")
    print(f"Reading images from {images_path}")

    cameras = read_cameras_txt(cameras_path)
    images = read_images_txt(images_path)

    transforms = create_transforms_json(cameras, images)
    output_path = '/vol/bitbucket//kt1923/4DGaussians/data/multipleview/office_0/colmap/sparse/0/transforms.json'

    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)

if __name__ == '__main__':
    main()