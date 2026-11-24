import json
import random
import os
import hashlib

random.seed(42)

# 1. 定义 16 个跨域工具的精简 Schema 协议（模拟真实企业多域 Agent）
TOOLS_SPEC = (
    "电商域: query_order(order_id), cancel_order(order_id, reason), refund_order(order_id, amount, reason), check_inventory(sku_id, city); "
    "办公域: create_event(title, time, attendees), cancel_event(event_id), send_email(to, subject, body), search_docs(keyword); "
    "运维域: check_server_cpu(host), restart_service(service_name, env), query_logs(service_name, level, limit); "
    "通用域: web_search(query), get_weather(city, date), calculate_expr(expression), ask_clarification(missing_params, prompt), none(reason)"
)

SYSTEM_PROMPT = f"你是企业级多域Agent调度引擎。可用工具：[{TOOLS_SPEC}]。请根据对话上下文输出严格合法的单行JSON（包含 tool_calls 列表或最终回复），禁止输出Markdown标记。"

# 2. 语料槽位池（用于组合生成高多样性、低模板感样本）
ORDERS = [f"ORD-{random.randint(10000, 99999)}" for _ in range(80)]
SKUS = [f"SKU-{random.randint(1000, 9999)}" for _ in range(60)]
CITIES = ["北京", "上海", "杭州", "深圳", "成都", "广州", "武汉", "南京", "西安", "苏州"]
REASONS = ["拍错了", "买重了", "不想要了", "发货太慢", "地址填错了", "价格降了"]
SERVICES = ["auth-service", "payment-gateway", "order-worker", "search-indexer", "redis-proxy"]
ENVS = ["prod", "staging", "dev"]
HOSTS = ["10.0.1.12", "10.0.1.45", "node-gpu-01", "db-master-02"]
EMAILS = ["zhangsan@corp.com", "hr@corp.com", "ops-team@corp.com", "pm@corp.com"]

def make_sample(messages):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}

def generate_single_turn_samples(n=450):
    samples = []
    for _ in range(n):
        task_type = random.choices(
            ["ecom", "office", "devops", "clarify", "reject", "hard_neg"],
            weights=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        )[0]

        if task_type == "ecom":
            sub = random.choice(["query", "cancel", "refund", "inv"])
            oid = random.choice(ORDERS)
            if sub == "query":
                q = random.choice([f"帮我查下 {oid} 的物流进度", f"订单 {oid} 发货了吗？", f"看看 {oid} 到哪了"])
                ans = {"tool_calls": [{"tool": "query_order", "arguments": {"order_id": oid}}]}
            elif sub == "cancel":
                r = random.choice(REASONS)
                q = random.choice([f"取消订单 {oid}，因为{r}", f"单号 {oid} 帮我关掉吧，{r}"])
                ans = {"tool_calls": [{"tool": "cancel_order", "arguments": {"order_id": oid, "reason": r}}]}
            elif sub == "refund":
                amt = random.choice([99.0, 199.5, 299.0, 49.9])
                r = random.choice(["商品破损", "质量问题", "七天无理由"])
                q = f"单号 {oid} 我要申请退款 {amt} 元，原因是{r}"
                ans = {"tool_calls": [{"tool": "refund_order", "arguments": {"order_id": oid, "amount": amt, "reason": r}}]}
            else:
                sku, city = random.choice(SKUS), random.choice(CITIES)
                q = random.choice([f"{city}仓还有 {sku} 的现货吗？", f"查一下 {sku} 在{city}有没有库存"])
                ans = {"tool_calls": [{"tool": "check_inventory", "arguments": {"sku_id": sku, "city": city}}]}

        elif task_type == "office":
            sub = random.choice(["event", "email", "docs"])
            if sub == "event":
                title = random.choice(["Q4架构评审", "算法周会", "Agent需求对齐", "发版复盘"])
                t = random.choice(["明天上午10点", "周五下午3点", "下周一14:30"])
                att = random.sample(EMAILS, 2)
                q = f"帮我预约一个{t}的【{title}】，参会人 {', '.join(att)}"
                ans = {"tool_calls": [{"tool": "create_event", "arguments": {"title": title, "time": t, "attendees": att}}]}
            elif sub == "email":
                to = random.choice(EMAILS)
                subj = random.choice(["周报同步", "压测报告", "权限申请"])
                q = f"给 {to} 发封邮件，主题是{subj}，内容写：请查收附件中的最新进展。"
                ans = {"tool_calls": [{"tool": "send_email", "arguments": {"to": to, "subject": subj, "body": "请查收附件中的最新进展。"}}]}
            else:
                kw = random.choice(["报销制度", "年假规定", "API鉴权规范", "安全红线"])
                q = f"帮我在公司内部文档里搜一下关于{kw}的说明"
                ans = {"tool_calls": [{"tool": "search_docs", "arguments": {"keyword": kw}}]}

        elif task_type == "devops":
            sub = random.choice(["cpu", "restart", "logs"])
            svc = random.choice(SERVICES)
            if sub == "cpu":
                host = random.choice(HOSTS)
                q = f"看一下机器 {host} 的 CPU 负载高不高"
                ans = {"tool_calls": [{"tool": "check_server_cpu", "arguments": {"host": host}}]}
            elif sub == "restart":
                env = random.choice(ENVS)
                q = f"把 {env} 环境的 {svc} 重启一下"
                ans = {"tool_calls": [{"tool": "restart_service", "arguments": {"service_name": svc, "env": env}}]}
            else:
                lvl = random.choice(["ERROR", "WARN"])
                lim = random.choice([20, 50, 100])
                q = f"拉取 {svc} 最近 {lim} 条 {lvl} 级别的日志"
                ans = {"tool_calls": [{"tool": "query_logs", "arguments": {"service_name": svc, "level": lvl, "limit": lim}}]}

        elif task_type == "clarify":
            case = random.choice([
                ("帮我取消一下订单", ["order_id", "reason"], "请提供需要取消的订单号及取消原因。"),
                (f"查一下 {random.choice(SKUS)} 还有没有货", ["city"], "请告知您要查询哪个城市的库存？"),
                (f"帮我重启一下 {random.choice(SERVICES)}", ["env"], "请确认需要重启的环境（prod / staging / dev）？"),
                ("给李总发个邮件说下午开会", ["to"], "请提供收件人的具体邮箱地址。")
            ])
            q = case[0]
            ans = {"tool_calls": [{"tool": "ask_clarification", "arguments": {"missing_params": case[1], "prompt": case[2]}}]}

        elif task_type == "hard_neg":
            # 区分 search_docs (内部制度) vs web_search (外部公开信息)
            if random.random() < 0.5:
                q = random.choice(["搜一下公司内网的差旅住宿标准", "查下内部知识库里的晋升答辩流程"])
                kw = "差旅住宿标准" if "差旅" in q else "晋升答辩流程"
                ans = {"tool_calls": [{"tool": "search_docs", "arguments": {"keyword": kw}}]}
            else:
                q = random.choice(["上网搜一下 PyTorch 2.6 的新特性", "帮我查下最新的英伟达财报新闻"])
                ans = {"tool_calls": [{"tool": "web_search", "arguments": {"query": q.replace("上网搜一下 ", "").replace("帮我查下", "")}}]}

        else: # reject
            q = random.choice([
                "帮我写一首关于春天的五言绝句",
                "讲个冷笑话听听",
                "Python 的 GIL 全局解释器锁是什么？",
                "推荐几部好看的科幻电影"
            ])
            ans = {"tool_calls": [{"tool": "none", "arguments": {"reason": "out_of_scope"}}]}

        samples.append(make_sample([
            {"role": "user", "content": q},
            {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}
        ]))
    return samples

def generate_parallel_samples(n=180):
    """生成并行多工具调用样本（Parallel Function Calling）"""
    samples = []
    for _ in range(n):
        oid = random.choice(ORDERS)
        sku, city = random.choice(SKUS), random.choice(CITIES)
        svc = random.choice(SERVICES)
        host = random.choice(HOSTS)

        pattern = random.choice([1, 2, 3])
        if pattern == 1:
            q = f"帮我查一下订单 {oid} 的状态，顺便看看{city}仓还有没有 {sku}"
            ans = {"tool_calls": [
                {"tool": "query_order", "arguments": {"order_id": oid}},
                {"tool": "check_inventory", "arguments": {"sku_id": sku, "city": city}}
            ]}
        elif pattern == 2:
            q = f"检查一下 {host} 的 CPU 情况，同时拉取 {svc} 最近 50 条 ERROR 日志"
            ans = {"tool_calls": [
                {"tool": "check_server_cpu", "arguments": {"host": host}},
                {"tool": "query_logs", "arguments": {"service_name": svc, "level": "ERROR", "limit": 50}}
            ]}
        else:
            q = f"查下{city}明天的天气，另外帮我在内部文档搜一下报销制度"
            ans = {"tool_calls": [
                {"tool": "get_weather", "arguments": {"city": city, "date": "明天"}},
                {"tool": "search_docs", "arguments": {"keyword": "报销制度"}}
            ]}
        samples.append(make_sample([
            {"role": "user", "content": q},
            {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}
        ]))
    return samples

def generate_multiturn_observation_samples(n=120):
    """生成多轮工具结果回填与总结样本（Observation -> Final Answer）"""
    samples = []
    for _ in range(n):
        oid = random.choice(ORDERS)
        city = random.choice(CITIES)
        sku = random.choice(SKUS)
        if random.random() < 0.5:
            q = f"帮我查下 {oid} 到哪了"
            call = json.dumps({"tool_calls": [{"tool": "query_order", "arguments": {"order_id": oid}}]}, ensure_ascii=False)
            obs = f"Tool Observation: {{\"order_id\": \"{oid}\", \"status\": \"派送中\", \"courier\": \"顺丰速运\"}}"
            final = f"您的订单 {oid} 目前正在由顺丰速运派送中，请保持手机畅通。"
        else:
            q = f"{city}还有 {sku} 吗？"
            call = json.dumps({"tool_calls": [{"tool": "check_inventory", "arguments": {"sku_id": sku, "city": city}}]}, ensure_ascii=False)
            obs = f"Tool Observation: {{\"sku_id\": \"{sku}\", \"city\": \"{city}\", \"stock\": 0}}"
            final = f"抱歉，商品 {sku} 在{city}仓库当前库存为 0，暂时缺货。"

        samples.append(make_sample([
            {"role": "user", "content": q},
            {"role": "assistant", "content": call},
            {"role": "user", "content": obs},
            {"role": "assistant", "content": final}
        ]))
    return samples

def generate_general_replay_samples(n=120):
    """混入 15% 通用指令数据，防止灾难性遗忘"""
    pairs = [
        ("用一句话解释什么是注意力机制（Attention）。", "注意力机制允许模型在处理序列时动态分配权重，聚焦于与当前预测最相关的上下文信息。"),
        ("解释 Python 中列表（list）和元组（tuple）的核心区别。", "列表是可变的（mutable），支持增删改操作；元组是不可变的（immutable），创建后无法修改且可作为字典键。"),
        ("将以下文本总结为一句话：今天下午两点全体研发在三楼会议室召开了关于大模型推理延迟优化的复盘会，确定了引入前缀缓存的方案。", "研发团队下午召开了推理延迟优化复盘会，决定引入前缀缓存方案。"),
        ("什么是 HTTP 状态码 502 Bad Gateway？", "502 表示作为网关或代理的服务器在尝试执行请求时，从上游服务器收到了无效的响应。")
    ]
    samples = []
    for i in range(n):
        q, a = pairs[i % len(pairs)]
        # 添加轻微指令变体
        prefix = random.choice(["", "请回答：", "简明扼要地说明：", "技术问答："])
        samples.append({
            "messages": [
                {"role": "system", "content": "你是一个专业、严谨的AI助手。"},
                {"role": "user", "content": prefix + q},
                {"role": "assistant", "content": a}
            ]
        })
    return samples

def deduplicate_and_validate(samples):
    """质量门禁：Schema 校验 + 哈希去重"""
    seen_hashes = set()
    clean = []
    for s in samples:
        # 校验最后一条 assistant 输出（如果是工具调用 JSON，验证其可解析性）
        last_msg = s["messages"][-1]["content"]
        if last_msg.startswith("{"):
            try:
                obj = json.loads(last_msg)
                assert "tool_calls" in obj and isinstance(obj["tool_calls"], list)
            except Exception:
                continue
        # 基于对话内容去重
        key = hashlib.md5(json.dumps(s["messages"][1:], ensure_ascii=False).encode()).hexdigest()
        if key not in seen_hashes:
            seen_hashes.add(key)
            clean.append(s)
    return clean

if __name__ == "__main__":
    raw_data = (
        generate_single_turn_samples(500) +
        generate_parallel_samples(200) +
        generate_multiturn_observation_samples(150) +
        generate_general_replay_samples(150)
    )
    clean_data = deduplicate_and_validate(raw_data)
    random.shuffle(clean_data)

    total = len(clean_data)
    n_train = int(total * 0.85)
    n_valid = int(total * 0.10)

    os.makedirs("data_pro", exist_ok=True)
    splits = {
        "train.jsonl": clean_data[:n_train],
        "valid.jsonl": clean_data[n_train:n_train+n_valid],
        "test.jsonl": clean_data[n_train+n_valid:]
    }

    for fname, rows in splits.items():
        with open(os.path.join("data_pro", fname), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ 已生成 data_pro/{fname}: {len(rows)} 条")
