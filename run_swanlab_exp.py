import os
import json
import time
import yaml
import argparse
import swanlab
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, train
from mlx_lm.tuner.datasets import load_dataset, CacheDataset
from mlx_lm.tuner.utils import linear_to_lora_layers

class SwanLabCallback(TrainingCallback):
    def on_train_loss_report(self, train_info: dict):
        step = train_info.get("iteration", 0)
        swanlab.log({
            "train/loss": train_info.get("train_loss", 0.0),
            "train/learning_rate": train_info.get("learning_rate", 0.0),
            "perf/tokens_per_sec": train_info.get("tokens_per_sec", 0.0),
        }, step=step)

    def on_val_loss_report(self, val_info: dict):
        step = val_info.get("iteration", 0)
        swanlab.log({
            "val/loss": val_info.get("val_loss", 0.0)
        }, step=step)

def run_experiment(exp_name: str, config_path: str, adapter_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    swanlab.init(
        project="EndSide-Agent-LoRA-Routing",
        experiment_name=exp_name,
        description="基于 Apple MLX 原生回调的端侧 Agent QLoRA 消融实验",
        config=cfg,
        mode="local"
    )

    model, tokenizer = load(cfg["model"])
    model.freeze()
    linear_to_lora_layers(model, cfg.get("num_layers", 16), cfg["lora_parameters"])

    args_ns = argparse.Namespace(
        data=cfg["data"],
        train=True,
        test=False,
        mask_prompt=cfg.get("mask_prompt", True)
    )
    train_set, valid_set, _ = load_dataset(args_ns, tokenizer)

    os.makedirs(adapter_path, exist_ok=True)
    training_args = TrainingArgs(
        batch_size=cfg.get("batch_size", 4),
        iters=cfg.get("iters", 300),
        val_batches=cfg.get("val_batches", 10),
        steps_per_report=cfg.get("steps_per_report", 15),
        steps_per_eval=cfg.get("steps_per_eval", 30),
        steps_per_save=cfg.get("save_every", 60),
        adapter_file=f"{adapter_path}/adapters.safetensors",
        max_seq_length=cfg.get("max_seq_length", 2048),
        grad_checkpoint=cfg.get("grad_checkpoint", False)
    )

    optimizer = optim.Adam(learning_rate=float(cfg.get("learning_rate", 1e-4)))

    print(f"🚀 启动训练并实时写入 SwanLab: [{exp_name}]")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        training_callback=SwanLabCallback()
    )

    # 保存适配器配置以便后续加载推理
    with open(f"{adapter_path}/adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_layers": cfg.get("num_layers", 16),
            "lora_parameters": cfg["lora_parameters"]
        }, f, indent=2)

    print("\n🧪 正在执行端到端业务指标评测...")
    eval_model, eval_tok = load(cfg["model"], adapter_path=adapter_path)
    test_path = os.path.join(cfg["data"], "test.jsonl")
    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f][:40]

    json_ok, route_ok, total_tool = 0, 0, 0
    latencies = []
    preview_rows = []

    for s in test_samples:
        msgs = s["messages"]
        target = msgs[-1]["content"]
        if not target.startswith("{"):
            continue
        total_tool += 1
        prompt = eval_tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        out = generate(eval_model, eval_tok, prompt=prompt, max_tokens=160, verbose=False).strip()
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        exp_tools = [c["tool"] for c in json.loads(target)["tool_calls"]]
        is_correct = False
        try:
            pred_tools = [c["tool"] for c in json.loads(out).get("tool_calls", [])]
            json_ok += 1
            if pred_tools == exp_tools:
                is_correct = True
                route_ok += 1
        except Exception:
            pass

        if len(preview_rows) < 6:
            preview_rows.append(f"**输入**: {msgs[-2]['content']}\n**输出**: `{out}` ({'✅' if is_correct else '❌'} | {lat:.1f}ms)")

    swanlab.log({
        "eval/json_valid_rate": (json_ok / total_tool * 100) if total_tool else 0,
        "eval/tool_routing_acc": (route_ok / total_tool * 100) if total_tool else 0,
        "eval/avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0,
        "eval/sample_predictions": swanlab.Text("\n\n---\n\n".join(preview_rows))
    })

    swanlab.finish()
    print(f"✅ 实验 [{exp_name}] 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.name, args.config, args.adapter)
