import itertools
import os
import json
import matplotlib.pyplot as plt

# 设定搜索空间
token_mixer_dims = [256, 512, 1024]
channel_mixer_dims = [256, 512, 1024]

# 训练参数
lr = 0.0001
optimizers = ["adamw"]
batch_sizes = [128]
weight_decays = [1e-4]

# 结果存储路径
result_dir = "./results/mlpmixer_width"
os.makedirs(result_dir, exist_ok=True)

# 记录最佳模型
best_acc = 0
best_model = None
results_dict = {}

# 遍历所有超参数组合
for token_dim, channel_dim, opt, batch, wd in itertools.product(token_mixer_dims, channel_mixer_dims, optimizers, batch_sizes, weight_decays):
    
    config_path = f"./model_configs/mlpmixer_width/mlpmixer_width_token{token_dim}_channel{channel_dim}.json"
    logdir = f"{result_dir}/results_token{token_dim}_channel{channel_dim}_opt{opt}_batch{batch}_wd{wd}"

    # 运行训练命令
    command = f"python main.py --model mlpmixer_width --model_config {config_path} --logdir {logdir} --lr {lr} --optimizer {opt} --batch_size {batch} --weight_decay {wd} --visualize"
    
    print(f"🚀 Running: {command}")
    os.system(command)  

    # 检查训练结果
    result_path = os.path.join(logdir, "results.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
        
        # 获取最佳验证准确率
        max_valid_acc = max(results["valid_accs"])
        
        # 更新最佳模型
        if max_valid_acc > best_acc:
            best_acc = max_valid_acc
            best_model = {
                "logdir": logdir,
                "token_mixer_dim": token_dim,
                "channel_mixer_dim": channel_dim,
                "optimizer": opt,
                "batch_size": batch,
                "weight_decay": wd,
                "valid_acc": max_valid_acc
            }
        
        # 存储所有实验数据
        results_dict[f"Token{token_dim}_Channel{channel_dim}"] = results

# 输出最佳模型信息
if best_model:
    print("\n🎯 Best MLP-Mixer Model Found!")
    print(f"📌 Logdir: {best_model['logdir']}")
    print(f"🛠 Token Mixer Dim: {best_model['token_mixer_dim']}")
    print(f"🔗 Channel Mixer Dim: {best_model['channel_mixer_dim']}")
    print(f"⚡ Optimizer: {best_model['optimizer']}")
    print(f"📦 Batch Size: {best_model['batch_size']}")
    print(f"🔗 Weight Decay: {best_model['weight_decay']}")
    print(f"🏆 Best Validation Accuracy: {best_model['valid_acc']:.4f}")

    # 保存最佳模型信息到 JSON
    with open(os.path.join(result_dir, "best_model.json"), "w") as f:
        json.dump(best_model, f, indent=4)
    print(f"📄 Best model info saved to {result_dir}/best_model.json")

# ============================== #
# 📊 绘制所有实验对比图
# ============================== #
print("\n📊 Plotting results...")

# 确保绘图目录存在
plot_dir = os.path.join(result_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# 需要绘制的指标
metrics = ["train_accs", "valid_accs", "train_losses", "valid_losses"]
titles = ["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss"]
y_labels = ["Accuracy", "Accuracy", "Loss", "Loss"]

for metric, title, ylabel in zip(metrics, titles, y_labels):
    plt.figure(figsize=(10, 6))
    
    for label, data in results_dict.items():
        plt.plot(data[metric], label=label)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)  # ⬇ Legend 放在下方
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{title} vs Epochs")
    plt.grid()
    
    save_path = os.path.join(plot_dir, f"{metric}.png")
    plt.savefig(save_path, bbox_inches="tight")  # ⬇ bbox_inches 避免裁剪 legend
    print(f"✅ Saved: {save_path}")

print("\n🎉 Grid Search Completed! All plots saved.")
