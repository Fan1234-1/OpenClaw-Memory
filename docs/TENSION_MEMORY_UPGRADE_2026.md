# ToneSoul 張力記憶升級（2026-02）

## 目的
在不增加系統複雜度的前提下，讓 OpenClaw-Memory 更貼近語魂系統：
- 保留 Hybrid RAG（FAISS + BM25 + RRF）骨幹
- 加入可控、可解釋的「張力共振」
- 讓記憶不只回答事實，也能優先回收高衝突/高責任的語義脈絡

## 為何現在做
2025–2026 的記憶研究與開源趨勢一致指出：
- 記憶需要 lifecycle（不是只靠一次檢索）
- 長期對話評估不能只看字面事實，還要看隱含約束一致性
- 過度複雜的記憶系統不一定帶來更好效益，應先做可控的最小增量

## 最小可行升級（已實作）
1. 記憶寫入新增欄位
- `tension: float in [0,1]`
- `tags: list[str]`
- `kind: memory type`
- `wave: {uncertainty_shift, divergence_shift, risk_shift, revision_shift}`

2. 查詢新增張力訊號
- 查詢可帶 `query_tension: float in [0,1]`
- 在 RRF 後做輕量重加權，可選兩種模式：

```text
# resonance mode
resonance = 1 - abs(query_tension - memory_tension)
final_score = rrf_score * (1 + 0.20 * clamp(resonance, 0, 1))

# conflict mode
delta = abs(query_tension - memory_tension)
final_score = rrf_score * (1 + 0.20 * clamp(delta, 0, 1))
```

3. 查詢新增波動共振（AI-meaningful）
```text
distance = mean(abs(query_wave[i] - memory_wave[i])) over shared dimensions
resonance = 1 - distance
final_score = score_after_tension * (1 + 0.25 * clamp(resonance, 0, 1))
```

4. 安全邊界
- 不改動原始向量/BM25檢索流程
- 沒有 `tension` 的舊記憶不會壞（僅不加權）
- `tension` 非法值會被拒絕

## 對語魂系統的對應
- 語魂中的「張力」對應為可持久化 metadata（`tension`）
- 在召回階段用「共振」反映當前提問與過往內在衝突的對齊程度
- 維持「不要消除分歧，而是讓分歧可見」：
  不是硬刪低分記憶，而是以可解釋方式調整排序

## 推薦工作流
1. 重要審議結論寫入 `--learn`，附上 `--tension` 與 `--tag`
2. 需要高一致性回答時，在查詢加上 `--query-tension`
3. 定期將高張力記憶做人工 review，再沉澱成規範文件

## 指令範例
```bash
python ask_my_brain.py --learn "在高風險部署時，先保守降級再恢復" --tension 0.82 --tag safety --tag deployment
python ask_my_brain.py "部署策略" --query-tension 0.8 --top-k 3
python ask_my_brain.py --memory-file docs/TENSION_MEMORY_UPGRADE_2026.md --tension 0.78 --tag architecture --tag tonesoul
```

## 2025–2026 參考脈絡（精簡）
- Mem0（2025）: 長期記憶可顯著降延遲與 token 成本，且提升多類問題表現
  https://arxiv.org/abs/2504.19413
- MemOS（2025）: 把 memory 當作一等系統資源（representation / scheduling / governance）
  https://arxiv.org/abs/2507.03724
- SeCom（2025）: 記憶單位粒度與壓縮去噪會顯著影響 retrieval quality
  https://arxiv.org/abs/2502.05589
- LoCoMo-Plus（2026）: 評估要覆蓋隱含約束一致性，不只 factual recall
  https://arxiv.org/abs/2602.10715

相關開源：
- Mem0: https://github.com/mem0ai/mem0
- Letta: https://github.com/letta-ai/letta
- LangMem: https://github.com/langchain-ai/langmem
- MemOS: https://github.com/MemTensor/MemOS

---
版本: v0.2
日期: 2026-02-27
