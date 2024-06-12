
#!/bin/bash #added by myself
#"shebang line": this line tells the system to use the Bash shell to execute the script. It is typically included at the top of shell scripts

#define and export Environment variables
workdir=$1
datatype=$2 # blender, hypernerf, llff
export CUDA_VISIBLE_DEVICES=0 #restricts the script to use only the first GPU 

#remove existing directories that contains intermediate data or results from previous COLMAP runs so that a fresh sparse reconstruction is generated in the current run
#remove also images prepared for COLMAP processing such that any old or partially processed images are removed
rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
python scripts/"$datatype"2colmap.py $workdir #runs a script named based on the 'datatype' variable (i.e. 'blender2colomap.py'). the script converts the dataset into a ormat that COLMAP can porcess
rm -rf $workdir/colmap
rm -rf $workdir/colmap/sparse/0

mkdir $workdir/colmap
cp -r $workdir/image_colmap $workdir/colmap/images
cp -r $workdir/sparse_ $workdir/colmap/sparse_custom
colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
python database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt
colmap exhaustive_matcher --database_path $workdir/colmap/database.db
mkdir -p $workdir/colmap/sparse/0

colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1

mkdir -p $workdir/colmap/dense/workspace
colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/workspace
colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply
