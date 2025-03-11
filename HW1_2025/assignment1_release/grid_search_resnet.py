import itertools
import os
import json

lr = 0.0001
optimizers = ["sgd", "momentum", "adam", "adamw"]  
batch_sizes = [64, 128]  
weight_decays = [1e-4, 5e-4]  

grid = list(itertools.product(optimizers, batch_sizes, weight_decays))

best_acc = 0
best_model = None

for opt, batch, wd in grid:
    logdir = f"./results/resnet/Q5/results_resnet18_lr{lr}_opt{opt}_batch{batch}_wd{wd}"
    
    momentum_flag = "--momentum 0.9" if opt in ["momentum", "sgd"] else ""

    command = f"python main.py --model resnet18 --model_config ./model_configs/resnet18.json --logdir {logdir}  --lr {lr} --optimizer {opt} --batch_size {batch} --weight_decay {wd} {momentum_flag} --visualize"
    
    print(f"Running: {command}")
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
                "optimizer": opt,
                "batch_size": batch,
                "weight_decay": wd,
                "valid_acc": max_valid_acc
            }

# 输出最佳模型信息
if best_model:
    print("\n🎯 Best ResNet18 Model Found!")
    print(f"📌 Logdir: {best_model['logdir']}")
    print(f"⚡ Optimizer: {best_model['optimizer']}")
    print(f"🛠 Batch Size: {best_model['batch_size']}")
    print(f"🔗 Weight Decay: {best_model['weight_decay']}")
    print(f"🏆 Best Validation Accuracy: {best_model['valid_acc']:.4f}")
