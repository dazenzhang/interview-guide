"""
MiniMax base_url + api_key 连通性 + Embedding 维度探测脚本

用法（cmd）:
  set MINIMAX_API_KEY=eyJxxxxxxxx
  set MINIMAX_BASE_URL=https://api.minimax.chat/v1
  python test_minimax.py
"""
import os
import sys
import json
import time
import textwrap

try:
    import requests
except ImportError:
    print("[ERROR] 缺少 requests 库。请先运行: pip install requests", file=sys.stderr)
    sys.exit(1)

BASE_URL = (os.environ.get("MINIMAX_BASE_URL") or "").rstrip("/")
API_KEY = os.environ.get("MINIMAX_API_KEY") or ""

if not BASE_URL:
    print("[ERROR] 请先设置环境变量 MINIMAX_BASE_URL，例如:", file=sys.stderr)
    print("  set MINIMAX_BASE_URL=https://api.minimax.chat/v1", file=sys.stderr)
    sys.exit(1)
if not API_KEY:
    print("[ERROR] 请先设置环境变量 MINIMAX_API_KEY", file=sys.stderr)
    sys.exit(1)

TIMEOUT = 15
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def short(text, limit=400):
    if text is None:
        return ""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def post(path, body):
    url = f"{BASE_URL}{path}"
    print(f"\n>>> POST {url}")
    print(f"    body = {short(json.dumps(body, ensure_ascii=False))}")
    t0 = time.time()
    try:
        resp = requests.post(url, headers=HEADERS, json=body, timeout=TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"    [EXC] {type(e).__name__}: {e}")
        return None
    dt = (time.time() - t0) * 1000
    print(f"    HTTP {resp.status_code}  ({dt:.0f} ms)")
    print(f"    body = {short(resp.text)}")
    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            return None
    return None


def test_chat():
    print("\n" + "=" * 60)
    print(" [1/2] Chat Completions 连通性测试")
    print("=" * 60)
    body = {
        "model": "MiniMax-Text-01",
        "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
        "max_tokens": 8,
    }
    candidates = [
        "/v1/chat/completions",
        "/text/chatcompletion_v2",
    ]
    for path in candidates:
        result = post(path, body)
        if result and "choices" in result:
            msg = result["choices"][0].get("message", {}).get("content", "")
            print(f"    [OK ] chat 命中 {path}: {short(msg, 80)}")
            return path
    print("    [FAIL] 所有 chat 路径都未返回 200，请检查 base_url / api_key / 模型名")
    return None


def test_embedding():
    print("\n" + "=" * 60)
    print(" [2/2] Embeddings 维度探测")
    print("=" * 60)
    candidates = [
        "/embeddings",
        "/v1/embeddings",
        "/v1/embedding/create",
        "/embedding/create",
    ]
    found_path = None

    # 阶段 A：探测 endpoint（先尝试带 type 字段）
    print("\n -- 阶段 A: 探测 endpoint（带 type 字段）--")
    for path in candidates:
        # 先用 type=db 试一次，命中即可
        result = post(path, {"model": "embo-01", "type": "db", "texts": ["ping"]})
        if result is None:
            continue
        if extract_dim(result):
            found_path = path
            break
        # 拿到 200 但维度拿不到，看下错误提示（可能是 type 值不对）
        base = result.get("base_resp") if isinstance(result, dict) else None
        if base:
            print(f"    [HINT] {path} 返回: {base.get('status_msg')}")
            found_path = path
            break

    if not found_path:
        print("    [FAIL] 没找到可用 embedding endpoint")
        return None, None, None

    # 阶段 B：在确认 endpoint 上，跑 6 种组合
    print(f"\n -- 阶段 B: 维度/参数矩阵探测 ({found_path}) --")
    rows = []
    for type_val in ("db", "query"):
        for dims in (None, 1024):
            body = {"model": "embo-01", "type": type_val, "texts": ["ping"]}
            if dims is not None:
                body["dimensions"] = dims
            label = f"type={type_val:5s} dims={str(dims):4s}"
            result = post(found_path, body)
            if result is None:
                rows.append((label, "HTTP 失败"))
                continue
            base = result.get("base_resp") if isinstance(result, dict) else None
            if base and base.get("status_code", 0) != 0:
                rows.append((label, f"ERR: {base.get('status_msg')}"))
                continue
            dim = extract_dim(result)
            rows.append((label, f"OK 维度={dim}" if dim else "OK 但无向量"))

    print("\n    探测矩阵:")
    for label, outcome in rows:
        print(f"      {label:24s} -> {outcome}")

    # 汇总：取第一个 OK 且非默认的作为最终结果
    default_dim = None
    down_dim = None
    for label, outcome in rows:
        if "OK 维度=" not in outcome:
            continue
        dim = int(outcome.split("=")[1])
        if "dims=1024" in label:
            down_dim = dim
        elif "dims=None" in label or "dims=None" in label:
            if default_dim is None:
                default_dim = dim

    return found_path, default_dim, down_dim


def extract_dim(result):
    if not isinstance(result, dict):
        return None
    # MiniMax 原生结构: {"vectors": [[...]], "base_resp": {...}}
    vectors = result.get("vectors")
    if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
        return len(vectors[0])
    # OpenAI 兼容结构: {"data": [{"embedding": [...]}]}
    data = result.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        vec = data[0].get("embedding") or data[0].get("vector")
        if isinstance(vec, list):
            return len(vec)
    return None


def main():
    print("MiniMax 连通性 + 维度探测")
    print(f"  base_url = {BASE_URL}")
    print(f"  api_key  = {API_KEY[:4]}***{API_KEY[-4:] if len(API_KEY) > 8 else '***'}")

    chat_path = test_chat()
    embed_path, default_dim, down_dim = test_embedding()

    print("\n" + "=" * 60)
    print(" 结果汇总")
    print("=" * 60)
    print(f"  chat 路径:        {chat_path or '(失败)'}")
    print(f"  embed 路径:       {embed_path or '(失败)'}")
    print(f"  默认向量维度:     {default_dim or '(未拿到)'}")
    print(f"  dimensions=1024:  {down_dim if down_dim else '(未拿到 / 失败)'}")

    if default_dim and down_dim == 1024:
        print("\n  ★ 结论: embo-01 支持 dimensions=1024 降维")
        print("    -> 直接用方案 A：配置 embedding-dimensions: 1024，不动 DB")
    elif default_dim and down_dim is None:
        print(f"\n  ⚠ 结论: 默认 {default_dim} 维，且不支持 dimensions 降维")
        print("    -> 走方案 B：扩 vector 列到 1536，重建索引，重向量化")
    elif default_dim and default_dim == 1024:
        print("\n  ★ 结论: 默认就是 1024 维，直接可用")
    elif default_dim:
        print(f"\n  ⚠ 结论: 默认 {default_dim} 维，未探测到 1024 降维")
        print("    -> 走方案 B：扩 vector 列到该维度，重建索引，重向量化")
    else:
        print("\n  ? 结论: 把完整输出贴给我再分析")


if __name__ == "__main__":
    main()