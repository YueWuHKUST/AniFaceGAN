CUDA_VISIBLE_DEVICES=0 python render.py \
    --generator_file './checkpoints/generator.pth' \
    --deform_file './checkpoints/dif.pth' \
    --output_dir './multiview_imgs/' \
    --image_size 256 \
    --gen_points_threshold 1e-5