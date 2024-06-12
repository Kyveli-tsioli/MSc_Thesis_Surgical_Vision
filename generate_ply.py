import numpy as np
from plyfile import PlyData, PlyElement


def read_points3D_txt(file_path):
    points3D= []
    with open(file_path, 'r') as f:
        lines= f.readlines()
        for line in lines[3:]:
            parts= line.strip().split()
            xyz= list(map(float, parts[1:4]))
            rgb =list(map(int, parts[4:7]))
            points3D.append((*xyz, *rgb))
        return points3D
    
def write_ply(points3D, output_file):
    vertices = np.array(points3D, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_file)

def main():
    input_txt = 'data/multipleview/office_0/colmap/sparse/0/points3D.txt'
    output_ply = 'data/multipleview/office_0/colmap/sparse/0/points3D.ply'
    points3D = read_points3D_txt(input_txt)
    write_ply(points3D, output_ply)

if __name__ == '__main__':
    main()