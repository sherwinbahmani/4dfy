seed=0
gpu=0
exp_root_dir=/path/to

# Original configs used in paper with 80 GB GPU memory

# Stage 1
# python launch.py --config configs/fourdfy_stage_1.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing"

# Stage 2
# ckpt=/path/to/fourdfy_stage_1/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_2.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Stage 3
# ckpt=/path/to/fourdfy_stage_2/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Low memory configs for 24-48 GB GPU memory

# Stage 1
# python launch.py --config configs/fourdfy_stage_1_low_vram.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing"

# Stage 2
# ckpt=/path/to/fourdfy_stage_1_low_vram/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_2_low_vram.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Stage 3
# ckpt=/path/to/fourdfy_stage_2_low_vram/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3_low_vram.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt


### Alternatives
# Use VideoCrafter2 in stage 3

# ckpt=/path/to/fourdfy_stage_2/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3_vc.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# ckpt=/path/to/fourdfy_stage_2_low_vram/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3_low_vram_vc.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Use deformation based approach to preserve quality in dynamic stage

# Stage 1
# python launch.py --config configs/fourdfy_stage_1_low_vram_deformation.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing"

# Stage 2
# ckpt=/path/to/fourdfy_stage_1_low_vram_deformation/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_2_low_vram_deformation.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Stage 3: Low memory configs for 24-48 GB GPU memory
# ckpt=/path/to/fourdfy_stage_2_low_vram_deformation/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3_low_vram_vc_deformation.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt

# Stage 3
# ckpt=/path/to/fourdfy_stage_2_low_vram_deformation/a_panda_dancing@timestamp/ckpts/last.ckpt
# python launch.py --config configs/fourdfy_stage_3_vc_deformation.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a panda dancing" system.weights=$ckpt