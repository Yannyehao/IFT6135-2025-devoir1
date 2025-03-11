import json
import os
import itertools

# 设定搜索空间
token_mixer_dims = [256, 512, 1024]
channel_mixer_dims = [256, 512, 1024]

# 配置文件存储路径
config_dir = "./model_configs/mlpmixer_width"
os.makedirs(config_dir, exist_ok=True)

# 遍历所有组合
for token_dim, channel_dim in itertools.product(token_mixer_dims, channel_mixer_dims):
    config = {
        "num_classes": 10,
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 256,
        "num_blocks": 8,
        "token_mixer_dim": token_dim,
        "channel_mixer_dim": channel_dim,
        "drop_rate": 0.0,
        "activation": "gelu"
    }

    config_filename = f"mlpmixer_width_token{token_dim}_channel{channel_dim}.json"
    config_path = os.path.join(config_dir, config_filename)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ Generated: {config_path}")
