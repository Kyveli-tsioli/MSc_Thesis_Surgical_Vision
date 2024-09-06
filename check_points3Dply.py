import open3d as o3d

def load_and_check_ply(filepath):
    # Try to load the PLY file
    point_cloud = o3d.io.read_point_cloud(filepath)

    # Check if the point cloud is empty
    if point_cloud.is_empty():
        print("The point cloud is empty. The file may be incorrectly formatted or damaged.")
    else:
        print("Point cloud loaded successfully. It contains", len(point_cloud.points), "points.")

        # Optionally visualize the point cloud to confirm it looks correct
        o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # Replace 'path_to_your_ply_file.ply' with the path to your PLY file
    ply_file_path = '/vol/bitbucket/kt1923/4DGaussians/data/multipleview/ns_images2/colmap/sparse/0/points3D.ply'
    load_and_check_ply(ply_file_path)