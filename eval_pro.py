import os
import json
import time
from mlx_lm import load, generate

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pro", "test.jsonl")

def evaluate_pro(model_path, adapter_path=None, name="Model"):
    print(f"\n{'='*22} 正在运行多维评测: {name} {'='*22}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f]

    # 额外加入 6 条完全脱离合成模板的口语化 OOD（分布外）高难度测试题
    sys_prompt = test_samples[0]["messages"][0]["content"] if "16" not in test_samples[0]["messages"][0]["content"] else ""
    for s in test_samples:
        if "企业级多域Agent调度引擎" in s["messages"][0]["content"]:
            sys_prompt = s["messages"][0]["content"]
            break

    ood_cases = [
        {"type": "OOD单工具", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "那个订单ORD-99887到底啥时候能送到啊，帮我瞅一眼"}], "expected_tools": ["query_order"]},
        {"type": "OOD并行调用", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "帮我看看prod环境下的payment-gateway最近20条ERROR日志，顺便查下10.0.1.99的CPU负载"}], "expected_tools": ["query_logs", "check_server_cpu"]},
        {"type": "OOD困难负样本", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "我想看看公司内部关于API鉴权规范的文档是怎么写的"}], "expected_tools": ["search_docs"]},
        {"type": "OOD缺参追问", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "麻烦帮我把线上服务重启一下"}], "expected_tools": ["ask_clarification"]},
        {"type": "OOD越界拒识", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "帮我写一段快速排序的Python代码"}], "expected_tools": ["none"]},
        {"type": "通用防遗忘", "messages": [{"role": "system", "content": "你是一个专业、严谨的AI助手。"}, {"role": "user", "content": "请简述Transformer中Self-Attention的作用。"}], "expected_keyword": "注意力"}
    ]

    stats = {
        "single": [0, 0],     # [正确数, 总数]
        "parallel": [0, 0],
        "safety": [0, 0],     # 缺参反问 + 拒识
        "multiturn": [0, 0],  # 多轮工具结果总结
        "general": [0, 0],    # 通用防遗忘
        "json_valid": [0, 0]
    }
    latencies = []

    # 1. 评测测试集（抽样前 60 条覆盖全场景）
    for s in test_samples[:60]:
        msgs = s["messages"]
        target = msgs[-1]["content"]
        input_msgs = msgs[:-1]
        prompt = tokenizer.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)

        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=160, verbose=False).strip()
        latencies.append((time.perf_counter() - t0) * 1000)

        # 判断样本类别
        if not target.startswith("{"):
            # 多轮自然语言总结 或 通用问答
            is_gen = "AI助手" in msgs[0]["content"]
            cat = "general" if is_gen else "multiturn"
            stats[cat][1] += 1
            if len(out) > 5 and not out.startswith("{"):
                stats[cat][0] += 1
            continue

        # 工具调用 JSON 类
        stats["json_valid"][1] += 1
        exp_obj = json.loads(target)
        exp_tools = [c["tool"] for c in exp_obj["tool_calls"]]

        if len(exp_tools) > 1:
            cat = "parallel"
        elif exp_tools[0] in ("ask_clarification", "none"):
            cat = "safety"
        else:
            cat = "single"
        stats[cat][1] += 1

        try:
            pred_obj = json.loads(out)
            stats["json_valid"][0] += 1
            pred_tools = [c["tool"] for c in pred_obj.get("tool_calls", [])]
            if pred_tools == exp_tools:
                stats[cat][0] += 1
        except Exception:
            pass

    # 2. 评测 OOD 分布外泛化样本
    ood_pass = 0
    for case in ood_cases:
        prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=160, verbose=False).strip()
        if "expected_keyword" in case:
            if len(out) > 10 and not out.startswith("{"):
                ood_pass += 1
        else:
            try:
                pred_tools = [c["tool"] for c in json.loads(out).get("tool_calls", [])]
                if pred_tools == case["expected_tools"]:
                    ood_pass += 1
            except Exception:
                pass

    def pct(pair):
        return f"{pair[0]}/{pair[1]} ({pair[0]/pair[1]*100:.1f}%)" if pair[1] > 0 else "N/A"

    print(f"📊 [{name}] 工业级多维评测报告:")
    print(f"  1. JSON 协议合法率       : {pct(stats['json_valid'])}")
    print(f"  2. 16域单工具路由准确率  : {pct(stats['single'])}")
    print(f"  3. 并行多工具调用准确率  : {pct(stats['parallel'])}")
    print(f"  4. 缺参追问与越界拒识率  : {pct(stats['safety'])}")
    print(f"  5. 多轮Observation总结率 : {pct(stats['multiturn'])}")
    print(f"  6. 通用能力防遗忘保持率  : {pct(stats['general'])}")
    print(f"  7. 分布外口语化(OOD)通过 : {ood_pass}/{len(ood_cases)} ({ood_pass/len(ood_cases)*100:.1f}%)")
    print(f"  8. 平均端到端响应延迟    : {sum(latencies)/len(latencies):.1f} ms")

if __name__ == "__main__":
    evaluate_pro("./qwen-3b", adapter_path=None, name="Base Model (微调前)")
    evaluate_pro("./qwen-3b", adapter_path="./adapters-pro-lora", name="Pro-LoRA (16域+并行+多轮)")
