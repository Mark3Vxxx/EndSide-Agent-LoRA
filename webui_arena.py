import json
import time
import gradio as gr
from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "你是企业级多域Agent调度引擎。可用工具：["
    "电商域: query_order(order_id), cancel_order(order_id, reason), refund_order(order_id, amount, reason), check_inventory(sku_id, city); "
    "办公域: create_event(title, time, attendees), cancel_event(event_id), send_email(to, subject, body), search_docs(keyword); "
    "运维域: check_server_cpu(host), restart_service(service_name, env), query_logs(service_name, level, limit); "
    "通用域: web_search(query), get_weather(city, date), calculate_expr(expression), ask_clarification(missing_params, prompt), none(reason)"
    "]。请根据对话上下文输出严格合法的单行JSON（包含 tool_calls 列表或最终回复），禁止输出Markdown标记。"
)

print("⏳ 正在加载底座模型与 Pro-LoRA 适配器，请稍候...")
base_model, tokenizer = load("./qwen-3b")
lora_model, _ = load("./qwen-3b", adapter_path="./adapters-pro-lora")
print("✅ 双引擎加载完成！")

# 模拟企业后端 16 域工具执行沙箱
def mock_execute_tool(tool_name: str, args: dict) -> str:
    handlers = {
        "query_order": lambda a: f"📦 [订单中心] 单号 {a.get('order_id')}：已到达【上海转运中心】，预计今日送达。",
        "cancel_order": lambda a: f"🚫 [订单中心] 单号 {a.get('order_id')} 已取消（原因：{a.get('reason')}）。",
        "refund_order": lambda a: f"💰 [财务退款] 单号 {a.get('order_id')} 已发起退款 ¥{a.get('amount')}（{a.get('reason')}）。",
        "check_inventory": lambda a: f"🏭 [WMS库存] {a.get('city')}仓 {a.get('sku_id')} 当前可用库存：38 件。",
        "create_event": lambda a: f"📅 [日历系统] 已创建会议《{a.get('title')}》，时间：{a.get('time')}，已通知 {a.get('attendees')}。",
        "search_docs": lambda a: f"📚 [内部知识库] 检索到《{a.get('keyword')}》相关规范文档 3 篇（已匹配最新版本）。",
        "check_server_cpu": lambda a: f"🖥️ [监控系统] 节点 {a.get('host')} 当前 CPU 负载：23.4%（健康）。",
        "restart_service": lambda a: f"🔄 [发布系统] 已在 {a.get('env')} 环境平滑重启服务 {a.get('service_name')}。",
        "query_logs": lambda a: f"📜 [日志中心] 拉取 {a.get('service_name')} 最近 {a.get('limit')} 条 {a.get('level')} 日志：发现 2 条超时告警。",
        "web_search": lambda a: f"🌐 [搜索引擎] 已检索关于“{a.get('query')}”的最新全网资讯。",
        "get_weather": lambda a: f"🌤️ [气象服务] {a.get('city')}{a.get('date')}：晴转多云，18°C ~ 25°C。"
    }
    if tool_name in handlers:
        return handlers[tool_name](args)
    return f"⚙️ [通用沙箱] 已执行 {tool_name}，参数：{args}"

def run_single_engine(model, user_query: str, is_lora: bool):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.perf_counter()
    raw_out = generate(model, tokenizer, prompt=prompt, max_tokens=180, verbose=False).strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    out_tokens = len(tokenizer.encode(raw_out))
    tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

    # 校验 JSON 协议与执行沙箱
    status_badge = ""
    exec_logs = []
    final_reply = ""

    try:
        parsed = json.loads(raw_out)
        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            status_badge = f"✅ **JSON 协议校验通过** | ⏱️ 延迟: `{latency_ms:.1f} ms` | ⚡ 速度: `{tps:.1f} tok/s`"
            observations = []
            for call in parsed["tool_calls"]:
                t_name = call.get("tool", "unknown")
                t_args = call.get("arguments", {})
                if t_name == "ask_clarification":
                    exec_logs.append(f"💬 **[触发缺参反问]** 缺失字段: `{t_args.get('missing_params')}`\n> 话术: {t_args.get('prompt')}")
                    final_reply = t_args.get("prompt", "请补充缺失参数。")
                elif t_name == "none":
                    exec_logs.append(f"🛡️ **[安全越界拒识]** 拦截非业务请求（原因: `{t_args.get('reason')}`），防止工具幻觉。")
                    final_reply = "该问题超出企业业务工具范围，已安全拦截并转交通用对话模型处理。"
                else:
                    obs = mock_execute_tool(t_name, t_args)
                    exec_logs.append(obs)
                    observations.append(obs)

            # 如果执行了真实工具，继续让微调模型做第二轮 Observation 总结
            if observations and is_lora:
                obs_text = "Tool Observation: " + " | ".join(observations)
                turn2_msgs = messages + [
                    {"role": "assistant", "content": raw_out},
                    {"role": "user", "content": obs_text}
                ]
                prompt2 = tokenizer.apply_chat_template(turn2_msgs, tokenize=False, add_generation_prompt=True)
                final_reply = generate(model, tokenizer, prompt=prompt2, max_tokens=120, verbose=False).strip()
            elif observations:
                final_reply = "\n".join(observations)
        else:
            status_badge = f"⚠️ **Schema 键名漂移 (缺少 tool_calls)** | ⏱️ `{latency_ms:.1f} ms`"
            exec_logs.append("❌ 下游路由解析器报错：JSON 顶层结构不符合 `tool_calls` 协议规范。")
            final_reply = "（解析失败，Agent 流程中断）"
    except json.JSONDecodeError:
        status_badge = f"❌ **JSON 解析崩溃 (SyntaxError)** | ⏱️ `{latency_ms:.1f} ms`"
        exec_logs.append("❌ 模型输出了非纯 JSON 文本（含废话或 Markdown 标记），`json.loads()` 抛出异常。")
        final_reply = "（解析失败，Agent 流程中断）"

    formatted_json = f"```json\n{raw_out}\n```"
    exec_md = "\n\n".join(exec_logs) if exec_logs else "无执行动作"
    return status_badge, formatted_json, exec_md, final_reply

def compare_models(user_query: str):
    if not user_query.strip():
        return ["请输入测试指令"] * 8
    base_res = run_single_engine(base_model, user_query, is_lora=False)
    lora_res = run_single_engine(lora_model, user_query, is_lora=True)
    return (*base_res, *lora_res)

with gr.Blocks(theme=gr.themes.Soft(), title="端侧 Agent LoRA 微调效果 A/B 竞技场") as demo:
    gr.Markdown(
        """
        # 🚀 端侧 Agent 路由引擎：QLoRA 微调前后 A/B 对比与执行追踪台
        **硬件底座**：Apple Silicon (24GB 统一内存) | **基座模型**：Qwen2.5-3B-Instruct (4-bit) | **微调策略**：16域指令 + 并行调用 + 多轮回填 + Prompt Loss Masking
        """
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="💬 输入 Agent 测试指令（支持单工具、并行多工具、缺参模糊指令、越界闲聊）",
            placeholder="例如：帮我查一下订单 ORD-88421 的状态，顺便看看杭州仓还有没有 SKU-7788",
            scale=4
        )
        run_btn = gr.Button("⚡ 同步运行双引擎对比", variant="primary", scale=1)

    gr.Examples(
        examples=[
            ["帮我查一下单号 ORD-88421 的物流到哪了"],
            ["帮我查一下订单 ORD-88421 的状态，顺便看看杭州仓还有没有 SKU-7788"],
            ["检查一下 10.0.1.12 的 CPU 负载，同时拉取 payment-gateway 最近 20 条 ERROR 日志"],
            ["帮我取消一下订单"],
            ["帮我写一首关于秋天的五言绝句"]
        ],
        inputs=query_input,
        label="🎯 点击一键填充高难度测试案例（直接体验对比效果）"
    )

    with gr.Row():
        # 左侧：微调前
        with gr.Column():
            gr.Markdown("### 🔴 Base Model（微调前裸模型）")
            base_status = gr.Markdown("等待运行...")
            base_json = gr.Markdown(label="原始输出")
            base_exec = gr.Markdown(label="沙箱解析结果")
            base_final = gr.Textbox(label="最终回复", interactive=False, lines=2)

        # 右侧：微调后
        with gr.Column():
            gr.Markdown("### 🟢 Pro-LoRA Agent（微调后路由引擎）")
            lora_status = gr.Markdown("等待运行...")
            lora_json = gr.Markdown(label="原始输出")
            lora_exec = gr.Markdown(label="沙箱执行轨迹 (Observation)")
            lora_final = gr.Textbox(label="多轮汇总最终回复", interactive=False, lines=2)

    run_btn.click(
        fn=compare_models,
        inputs=query_input,
        outputs=[base_status, base_json, base_exec, base_final, lora_status, lora_json, lora_exec, lora_final]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
