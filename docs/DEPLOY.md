# 部署指南

把项目推到公网,HR 一点链接就能体验。**总耗时 30 分钟**(已熟悉云平台)/ **2 小时**(第一次)。

---

## 推荐组合:**Render(后端 + DB)+ Vercel(前端)**

两个免费档拼起来:
- Render 跑 FastAPI + Postgres
- Vercel 跑 React 静态站点(全球 CDN,加载快)

结果是两个 URL:
- `https://你的项目.onrender.com` — 后端 API
- `https://你的项目.vercel.app` — 用户访问的前端

---

## Step 1:推 GitHub

```bash
cd /Users/neo/enterprise-multimodal-copilot
git remote add origin https://github.com/<你的用户名>/enterprise-multimodal-copilot.git
git push -u origin main
```

⚠️ **确认 `.env` / `.env.local` 不在 commit 里**(应该已被 .gitignore)。`DASHSCOPE_API_KEY` 不能进 git。

---

## Step 2:Render 部署后端 + DB

1. 注册 https://render.com(GitHub OAuth 就行)
2. 仪表盘 → **New** → **Blueprint**
3. 选刚才的 repo → Render 自动识别项目里的 [render.yaml](render.yaml)
4. **Apply** — Render 会创建:
   - Web Service:`enterprise-rag-agent`(从 Dockerfile 构建,5-10 min 首次)
   - Postgres:`enterprise-rag-db`
5. Web Service → **Environment** → 加密钥:
   - `DASHSCOPE_API_KEY` = `sk-xxx`(你的 Qwen key)
6. 等到状态变绿 → 拿到后端 URL,例如 `https://enterprise-rag-agent.onrender.com`
7. 验证:`curl https://enterprise-rag-agent.onrender.com/health` 返回 200

**注意**:免费档**15 分钟无访问会休眠**,第一次访问冷启动 30 秒。如果想常驻,$7/月 starter 档。

---

## Step 3:Vercel 部署前端

1. 注册 https://vercel.com(GitHub OAuth)
2. **Add New** → **Project** → 选同一个 repo
3. **Root Directory**:改成 `frontend`(因为前端代码在子目录里)
4. **Framework Preset**:Vite(自动识别)
5. **Environment Variables**:
   - `VITE_API_BASE` = `https://enterprise-rag-agent.onrender.com`(Step 2 拿到的后端 URL)
6. **Deploy** → 等 1-2 分钟
7. 拿到前端 URL,例如 `https://enterprise-rag-agent-xyz.vercel.app`

---

## Step 4:回 Render 加 CORS 信任

后端默认只信任 `127.0.0.1:5714`(本地开发)。生产要把 Vercel 域名加进去:

1. Render → Web Service → Environment → 编辑 `COPILOT_CORS_ALLOW_ORIGINS`
2. 改成:`https://enterprise-rag-agent-xyz.vercel.app,https://enterprise-rag-agent-xyz-*.vercel.app`
3. Save → Render 自动重启

(Vercel 给 PR 分支生成预览 URL 也带 `*` 通配,所以要加 `*.vercel.app`。)

---

## Step 5:首次播种(可选,让 demo 数据可用)

第一次部署后,DB 是空的。可以从本地远程跑一次 seed:

```bash
# 拿到 Render Postgres 的连接串(Dashboard → Database → Connect → External URL)
export DATABASE_URL='postgresql://...'
./.venv/bin/python scripts/seed_db.py
```

或者打开后端的 `/health`,然后在本地用 `scripts/build_kb.py` 推 KB 数据(进阶,先不做)。

---

## 常见问题

### Render 部署失败,日志里 "no module named X"
检查 [Dockerfile](Dockerfile) 里 `pip install` 段是否包含你新加的库。新加的话,push 一次 commit 触发重建。

### 前端能加载,但所有 API 调用 CORS 失败
Step 4 没做对,或者 `VITE_API_BASE` 写错(末尾不能带 `/`)。

### "Application Error" / 502 Bad Gateway
- Render 冷启动中(免费档),等 30 秒刷新
- 看 Render → Logs:常见是 `DATABASE_URL` 未注入,或 Postgres 还没 ready
- alembic 迁移失败(看日志最早几行)

### Render 免费 DB 90 天后会过期
免费 Postgres 期限 90 天,过期后只能新建。如果做长期 demo,要么升级到 $7/月 paid,要么换 [Supabase](https://supabase.com)/[Neon](https://neon.tech) 永久免费档。

### 前端按钮调 API 慢得离谱
- 第一次访问冷启动是正常的(30 秒)
- 后续每个请求 100-300ms 是 Render 跨区延迟,正常
- 如果想要瞬时响应,后端升级到 $7 starter(常驻不休眠)

---

## 简历写法

```
Live demo: https://enterprise-rag-agent-xyz.vercel.app
Source:    https://github.com/<your-username>/enterprise-multimodal-copilot
```

让 HR 30 秒看简历时**直接点开就能玩**。这是简历最大的转化器。

---

## 替代方案对比

| 平台 | 后端 | DB | 前端 | 一句话 |
|---|---|---|---|---|
| **Render + Vercel** ⭐ | ✅ Docker / 免费档(休眠) | ✅ Postgres 90d 免费 | ✅ Vercel CDN | **简历首选** |
| Fly.io | ✅ Docker / $0 起 | ✅ Postgres 3GB 免费 | (静态) | 全平台一致,Fly 跑全栈也行 |
| Railway | ✅ 但要付费 | ✅ | ✅ | $5/月起步,DX 最好 |
| AWS / GCP | ✅ | ✅ | ✅ | 学习曲线陡,**不推荐第一次部署** |
| 自有 VPS | ✅ 全控 | 需自配 | 需自配 | 成本最低但你要会 nginx/systemd |
