import json
import time
from mlx_lm import load, generate

SYSTEM_PROMPT = "你是Agent路由引擎。可用工具：query_order(order_id), cancel_order(order_id, reason), check_inventory(sku_id, city), transfer_human(department)。严格输出单行纯JSON，禁止输出Markdown标记或任何解释。"

def evaluate(model_path, adapter_path=None, name="Baseline"):
    print(f"\n{'='*20} 正在评测: {name} {'='*20}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f]

    total = len(test_cases)
    json_valid = 0
    tool_correct = 0
    exact_match = 0
    latencies = []

    for i, case in enumerate(test_cases):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["query"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        t0 = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False).strip()
        latencies.append((time.perf_counter() - t0) * 1000)

        try:
            pred = json.loads(output)
            json_valid += 1
            exp = case["expected"]
            
            if pred.get("tool") == exp["tool"]:
                tool_correct += 1
                # 对于 ask_clarification，重点核对 missing_params 是否一致（忽略提示语措辞差异）
                if exp["tool"] == "ask_clarification":
                    if pred.get("arguments", {}).get("missing_params") == exp["arguments"]["missing_params"]:
                        exact_match += 1
                elif pred.get("arguments") == exp["arguments"]:
                    exact_match += 1
                else:
                    print(f"[参数偏差] 输入: {case['query']}\n  期望: {exp}\n  实际: {pred}")
            else:
                print(f"[路由错误] 输入: {case['query']}\n  期望: {exp}\n  实际: {pred}")
        except json.JSONDecodeError:
            print(f"[JSON解析失败] 输入: {case['query']}\n  原始输出: {output}")

    avg_lat = sum(latencies) / total
    print(f"\n📊 [{name}] 评测汇总 (共 {total} 条样本):")
    print(f"  - JSON 格式合法率 : {json_valid}/{total} ({json_valid/total*100:.1f}%)")
    print(f"  - 工具路由准确率  : {tool_correct}/{total} ({tool_correct/total*100:.1f}%)")
    print(f"  - 端到端完全匹配率: {exact_match}/{total} ({exact_match/total*100:.1f}%)")
    print(f"  - 平均端到端延迟  : {avg_lat:.1f} ms")

if __name__ == "__main__":
    evaluate("./qwen-3b", adapter_path=None, name="Base Model (微调前)")
    evaluate("./qwen-3b", adapter_path="./adapters-r4", name="LoRA Rank=4 (微调后)")
    evaluate("./qwen-3b", adapter_path="./adapters-r16", name="LoRA Rank=16 (微调后)")
