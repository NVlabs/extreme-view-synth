# The project folder must contain a folder "images" with all the images.
DATASET_PATH=$1

colmap feature_extractor \
   --database_path $DATASET_PATH/db.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.single_camera=1

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/db.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/db.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.init_min_tri_angle=0.1 \
    --Mapper.tri_min_angle=0.1 \
    --Mapper.filter_min_tri_angle=0.1 \
    --Mapper.init_max_forward_motion=1.0

mkdir -p $DATASET_PATH/dense/0

colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense/0 \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense/0 \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.min_triangulation_angle=0.1 \
    --PatchMatchStereo.filter_min_triangulation_angle=0.1

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense/0 \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
