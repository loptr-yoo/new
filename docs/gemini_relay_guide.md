# 通过第三方中转调用 Gemini 的最佳实践（基于 CloseAI 文档整理）

## 1. 接口与协议选择
- 推荐优先使用「原生 Gemini 协议」：功能更完整、更稳定
  - 基础域名（中转）：https://api.openai-proxy.org/google
  - 端点路径：/v1beta/models/{model}:generateContent
  - 鉴权：query 参数 ?key=YOUR_API_KEY
- OpenAI 兼容协议：仅在框架仅支持 OpenAI 格式时使用（稳定性较低）
  - 基础域名（中转）：https://api.openai-proxy.org/v1
  - 端点路径：/chat/completions
  - 鉴权：请求头 Authorization: Bearer YOUR_API_KEY

说明：两种协议的模型名一致（如 gemini-2.5-flash），但请求/响应格式不同。若无强需求，选用「原生 Gemini 协议」。

## 2. 环境变量与加载
建议在项目根目录的 .env.local 或后端 backend/.env.local 中配置以下变量：

```
# 必填：API Key（使用中转平台颁发的 key）
GEMINI_API_KEY=sk-xxxxx

# 可选：切换到中转域名（否则默认直连 Google 官方）
GEMINI_BASE_URL=https://api.openai-proxy.org/google

# 如需 OpenAI 兼容协议
OPENAI_BASE_URL=https://api.openai-proxy.org/v1
OPENAI_API_KEY=sk-xxxxx
```

后端会按顺序加载 backend/.env.local、backend/.env、根目录 .env.local、根目录 .env，并兼容 VITE_GEMINI_API_KEY 等命名。详见 backend/settings.py。

## 3. 请求头、请求体与响应
### 3.1 原生 Gemini 协议（非流式）
- Method：POST
- URL：{GEMINI_BASE_URL}/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}
- Headers：
  - Content-Type: application/json
  - Accept: application/json
- Body（示例）：
```
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Say Hello"}]
    }
  ],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 4000,
    "responseMimeType": "application/json"
  },
  "systemInstruction": {
    "parts": [{"text": "You are a helpful assistant"}]
  }
}
```
- 关键响应字段（简化）：
  - candidates[0].content.parts[0].text 为模型文本输出

### 3.2 OpenAI 兼容协议（流式/非流式）
- URL：{OPENAI_BASE_URL}/chat/completions
- Headers：
  - Content-Type: application/json
- Auth：Authorization: Bearer {OPENAI_API_KEY}
- 非流式 Body（示例）：
```
{
  "model": "gemini-2.5-flash",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "response_format": {"type": "json_object"},
  "temperature": 0.7,
  "max_tokens": 4000,
  "stream": false
}
```
- 流式 Body（stream=true）：
```
{
  "model": "gemini-2.5-flash",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```
- 流式响应使用 SSE，事件帧包含 data: 前缀。

## 4. 多语言调用示例
### 4.1 Python（httpx，原生 Gemini 协议）
```python
import os, httpx, asyncio

BASE = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

async def call_gemini(prompt: str):
    url = f"{BASE}/v1beta/models/{MODEL}:generateContent?key={KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048, "responseMimeType": "application/json"}
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10, read=30), follow_redirects=True, trust_env=True) as client:
        r = await client.post(url, json=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        return (((data or {}).get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text")

print(asyncio.run(call_gemini("Say Hello")))
```

### 4.2 Node.js（fetch，OpenAI 兼容流式）
```js
const BASE = process.env.OPENAI_BASE_URL || 'https://api.openai-proxy.org/v1';
const KEY = process.env.OPENAI_API_KEY;

async function streamCompat() {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${KEY}` },
    body: JSON.stringify({
      model: 'gemini-2.5-flash',
      messages: [{ role: 'user', content: 'Hello!' }],
      stream: true
    })
  });
  if (!res.ok || !res.body) throw new Error(await res.text());
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    console.log(chunk); // 解析 data: 行
  }
}
```

### 4.3 cURL（OpenAI 兼容非流式）
```bash
curl https://api.openai-proxy.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## 5. 强化请求与错误重试
### 5.1 后端通用请求函数（已在项目中实现）
- 位置：backend/core/llm/provider.py 的 `safe_fetch`
- 特性：
  - httpx 超时：connect=10s、read=30s，总 30s
  - follow_redirects=True，trust_env=True（自动尊重系统代理/证书）
  - 指数退避重试（0s/1.5s/3s），记录失败日志到 `llm.{provider}`
  - 非 2xx 状态抛出包含状态码与响应文本的错误

### 5.2 速率限制与 429 处理建议
- 若收到 429 或标准限流响应：读取 `Retry-After`（若存在）作为等待时间；否则按退避序列等待。
- 同一 key 建议在应用侧做令牌桶限速（如 per-model QPS 上限），结合队列或并发门控。

## 6. 与现有代码的对齐
- 基础域名可在运行时通过环境变量覆盖：
  - GEMINI_BASE_URL（默认 https://generativelanguage.googleapis.com）
  - OPENAI_BASE_URL（默认 https://api.openai-proxy.org/v1）
  - DEEPSEEK_BASE_URL（默认 https://api.deepseek.com）
- 原生 Gemini 协议：由 `GeminiClient` 调用 `/v1beta/models/{model}:generateContent?key=...`，响应解析 candidates[].content.parts[].text。
- OpenAI 兼容协议：由 `OpenAICompatibleClient` 调用 `/v1/chat/completions`，非流式解析 choices[0].message.content。

## 7. 调试与日志
- 后端已开启：控制台 + backend.log 双通道 INFO 级别日志，UTF-8 编码（见 backend/main.py）。
- 设置 `PYTHONUNBUFFERED=1` 确保日志实时输出。
- 如果网络失败，请检查：代理环境变量（HTTP_PROXY/HTTPS_PROXY/NO_PROXY）、证书信任链（企业根证书）、防火墙出站策略。

## 8. 最佳实践摘要
- 无强需求优先选择「原生 Gemini 协议」；仅在必须情况下使用 OpenAI 兼容协议。
- 所有敏感信息放入 .env.local 并已加入 .gitignore。
- 在请求层实现重试和限流；在业务层容错解析与结构校验（详见 response_parser.py）。

以上内容与现有代码完全对齐，可直接按照环境变量切换至 CloseAI 中转，或回退到官方域名运行。

