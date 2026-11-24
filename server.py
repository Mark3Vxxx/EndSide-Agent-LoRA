import copy
import json
import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

app = FastAPI(title="EndSide Agent Inference Gateway (MLX + Prefix Cache)")

SYSTEM_PROMPT = (
    "你是企业级多域Agent调度引擎。可用工具：["
    "电商域: query_order(order_id), cancel_order(order_id, reason), refund_order(order_id, amount, reason), check_inventory(sku_id, city); "
    "办公域: create_event(title, time, attendees), cancel_event(event_id), send_email(to, subject, body), search_docs(keyword); "
    "运维域: check_server_cpu(host), restart_service(service_name, env), query_logs(service_name, level, limit); "
    "通用域: web_search(query), get_weather(city, date), calculate_expr(expression), ask_clarification(missing_params, prompt), none(reason)"
    "]。请根据对话上下文输出严格合法的单行JSON（包含 tool_calls 列表或最终回复），禁止输出Markdown标记。"
)

print("⏳ 正在加载 4-bit 底座与 Pro-LoRA 适配器...")
model, tokenizer = load("./qwen-3b", adapter_path="./adapters-pro-lora")

# 构建并预填充 System Prompt 的前缀 KV Cache（Prefix Caching 核心优化）
sys_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
sys_prefix_str = tokenizer.apply_chat_template(sys_messages, tokenize=False, add_generation_prompt=False)
sys_prefix_tokens = tokenizer.encode(sys_prefix_str)

BASE_PREFIX_CACHE = make_prompt_cache(model)
# 执行一次前向预热填充前缀缓存
for _ in stream_generate(model, tokenizer, prompt=sys_prefix_tokens, max_tokens=1, prompt_cache=BASE_PREFIX_CACHE):
    pass
mx.eval([c.state for c in BASE_PREFIX_CACHE])
print(f"✅ Prefix KV Cache 预热完成！已固化 {len(sys_prefix_tokens)} 个系统提示词 Token。")

# MLX 统一内存推理锁（防止多线程并发操作 Metal 队列冲突）
inference_lock = asyncio.Lock()

class AgentRequest(BaseModel):
    query: str
    use_prefix_cache: bool = True
    max_tokens: int = 128

def _run_inference_sync(query: str, use_prefix_cache: bool, max_tokens: int):
    t_start = time.perf_counter()
    ttft_ms = 0.0
    output_text = ""
    token_count = 0

    if use_prefix_cache:
        # 深拷贝已预热的 System Prompt KV Cache，仅增量编码 User Turn
        cache = copy.deepcopy(BASE_PREFIX_CACHE)
        user_turn_str = f"\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenizer.encode(user_turn_str, add_special_tokens=False)
        gen_iter = stream_generate(model, tokenizer, prompt=prompt_tokens, max_tokens=max_tokens, prompt_cache=cache)
    else:
        # 传统全量重算模式（用于对比消融）
        full_msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
        full_prompt = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=True)
        gen_iter = stream_generate(model, tokenizer, prompt=full_prompt, max_tokens=max_tokens)

    for resp in gen_iter:
        if token_count == 0:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        output_text += resp.text
        token_count += 1

    total_ms = (time.perf_counter() - t_start) * 1000
    tpot_ms = (total_ms - ttft_ms) / max(token_count - 1, 1)

    # 校验 JSON 协议合法性
    json_valid = False
    try:
        json.loads(output_text.strip())
        json_valid = True
    except Exception:
        pass

    return {
        "output": output_text.strip(),
        "json_valid": json_valid,
        "metrics": {
            "use_prefix_cache": use_prefix_cache,
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "generated_tokens": token_count
        }
    }

@app.post("/v1/agent/route")
async def route_agent(req: AgentRequest):
    async with inference_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _run_inference_sync, req.query, req.use_prefix_cache, req.max_tokens
        )
        return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
