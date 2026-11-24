import asyncio
import time
import statistics
import httpx

API_URL = "http://127.0.0.1:8000/v1/agent/route"

TEST_QUERIES = [
    "帮我查一下单号 ORD-88421 的物流到哪了",
    "取消订单 ORD-55667，因为买重了",
    "查一下 SKU-7788 在杭州有没有库存",
    "帮我查一下订单 ORD-90123 的状态，顺便看看上海仓还有没有 SKU-3090",
    "检查一下 10.0.1.12 的 CPU 负载，同时拉取 payment-gateway 最近 20 条 ERROR 日志",
    "把 prod 环境的 auth-service 重启一下",
    "帮我在内部文档里搜一下报销制度",
    "帮我取消一下订单"
]

async def send_request(client: httpx.AsyncClient, query: str, use_cache: bool):
    t0 = time.perf_counter()
    resp = await client.post(API_URL, json={"query": query, "use_prefix_cache": use_cache}, timeout=60.0)
    e2e_ms = (time.perf_counter() - t0) * 1000
    data = resp.json()
    m = data["metrics"]
    return {
        "e2e_ms": e2e_ms,
        "ttft_ms": m["ttft_ms"],
        "tpot_ms": m["tpot_ms"],
        "tokens": m["generated_tokens"],
        "valid": data["json_valid"]
    }

async def run_benchmark(concurrency: int, total_requests: int, use_cache: bool):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        # 预热 1 次
        await send_request(client, TEST_QUERIES[0], use_cache)

        async def worker(i):
            async with sem:
                q = TEST_QUERIES[i % len(TEST_QUERIES)]
                res = await send_request(client, q, use_cache)
                results.append(res)

        t_start = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        wall_time_s = time.perf_counter() - t_start

    ttfts = sorted(r["ttft_ms"] for r in results)
    e2es = sorted(r["e2e_ms"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    valid_rate = sum(1 for r in results if r["valid"]) / len(results) * 100

    def percentile(arr, p):
        idx = min(int(len(arr) * p), len(arr) - 1)
        return arr[idx]

    mode_str = "开启 Prefix KV Cache" if use_cache else "关闭 Cache (全量重算)"
    print(f"\n📊 压测模式: [{mode_str}] | 并发数: {concurrency} | 总请求: {total_requests}")
    print(f"  - 首字延迟 TTFT (平均 / P95) : {statistics.mean(ttfts):.1f} ms / {percentile(ttfts, 0.95):.1f} ms")
    print(f"  - 端到端延迟 (P50 / P95 / P99): {percentile(e2es, 0.50):.1f} ms / {percentile(e2es, 0.95):.1f} ms / {percentile(e2es, 0.99):.1f} ms")
    print(f"  - 系统总吞吐 (QPS / Tok/s)   : {total_requests / wall_time_s:.2f} req/s | {total_tokens / wall_time_s:.1f} tok/s")
    print(f"  - 高并发下 JSON 协议合法率   : {valid_rate:.1f}%")

async def main():
    print("🚀 开始 AI Infra 推理网关性能压测...")
    # 对比 1：关闭 Prefix Cache（串行基准）
    await run_benchmark(concurrency=1, total_requests=16, use_cache=False)
    # 对比 2：开启 Prefix Cache（串行加速）
    await run_benchmark(concurrency=1, total_requests=16, use_cache=True)
    # 对比 3：开启 Prefix Cache（4 并发模拟线上多请求排队）
    await run_benchmark(concurrency=4, total_requests=24, use_cache=True)

if __name__ == "__main__":
    asyncio.run(main())
