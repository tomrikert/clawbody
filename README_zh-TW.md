# 🦞🤖 ClawBody (優化版)

**為您的 OpenClaw AI 代理提供具有表現力的具身實體！**

[English Version](README.md)

ClawBody 彌合了高層級 AI 智慧 (OpenClaw) 與底層機器人控制 (Reachy Mini) 之間的鴻溝。透過利用 OpenAI 的 Realtime API，它建立了一個超低延遲的語音對話循環，讓您的 AI 助手 Clawson 能夠在現實世界中看、聽並以物理方式表達情感。

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

---

## 🚀 重大技術更新 (2026年2月)

本專案近期進行了顯著的架構升級，旨在強化「具身智慧 (Embodied Intelligence)」：

- **自然具身表現 (Natural Embodiment)**：引入了 **「轉向級自然手勢 (Natural Turn-Level Gestures)」** 與 **「語音同步身體搖擺 (Speech-Synced Body Sway)」**。機器人在說話時不再僵硬，而是會根據對話的節奏與情緒進行流暢的物理律動。
- **動態能力發現 (Dynamic Capability Discovery)**：全新的 **「能力註冊機制 (Capability Registry)」** 會自動掃描 Reachy daemon 中所有預錄的情感與舞蹈。這使得 AI 能夠在不修改代碼的情況下，自動發現並運用機器人的新行為。
- **上下文感知觸發 (Context-Aware Triggers)**：新增 **「關鍵字手勢 (Cue Word Gestures)」** 支援。機器人現在能根據即時逐字稿中檢測到的特定詞彙，精準地觸發對應的動作。
- **感知優化**：優化了 MediaPipe 追蹤數據格式，並解決了 OpenClaw Gateway 的 CORS 連線問題，確保即時視覺循環更加穩定。

---

## ✨ 核心特色

- **👁️ 智慧眼神接觸**：25Hz 即時人臉追蹤 (MediaPipe/YOLO)，確保機器人始終與使用者保持互動感。
- **🎭 表現力手勢**：自動化手勢與語音輸出同步，包含用於強調語氣的「誇張宏觀手勢」。
- **🧠 OpenClaw 深度整合**：透過實體人格化身，完整調用 OpenClaw 工具（行事曆、智慧家居、網頁搜索）。
- **💃 情感引擎**：可動態觸發圖書館中任何預錄的情感或舞蹈。
- **🎤 低延遲語音**：由 OpenAI Realtime API 驅動，實現如人類般的自然反應速度。
- **🖥️ 模擬器優先**：完整支援 MuJoCo 模擬，無需實體硬體即可進行開發。

---

## 🏗️ 系統架構

ClawBody 作為以下三大系統的協調者：
1. **OpenAI Realtime API**：處理音訊流並生成低延遲回覆。
2. **OpenClaw Gateway**：提供人格、記憶與工具調用能力的「大腦」。
3. **Reachy Mini SDK/Daemon**：控制機器人伺服馬達與感測器的「神經系統」。

---

## 🚀 快速開始

### 基本需求
- Python 3.11+
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Gateway](https://github.com/openclaw/openclaw)
- OpenAI API 金鑰 (需具備 Realtime API 權限)

### 選項 A：模擬器安裝步驟

```bash
git clone https://github.com/dAAAb/clawbody
cd clawbody
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# 配置 .env 環境變數
cp .env.example .env

# 終端機 1：啟動模擬器
reachy-mini-daemon --sim

# 終端機 2：執行 ClawBody
clawbody --gradio
```

### 選項 B：實體機器人安裝步驟

```bash
# SSH 登入機器人
ssh pollen@reachy-mini.local

# 在 app 環境中複製並安裝
git clone https://github.com/dAAAb/clawbody
cd clawbody
/venvs/apps_venv/bin/pip install -e .

# 直接在硬體上執行
clawbody
```

---

## 🛠️ 機器人能力 (AI 可調用)

| 能力 | 技術細節 |
|------------|-------------------|
| **自然手勢** | 根據逐字稿變動量同步觸發轉向級動作 |
| **情感註冊** | 自動掃描並註冊 daemon 中預錄的表情 |
| **人臉追蹤** | 使用 25Hz 視覺數據進行 PID 控制的頭部運動 |
| **視覺描述** | 捕捉畫面並利用 GPT-4o-mini 進行場景理解 |

---

## 📄 授權資訊

本專案採用 Apache 2.0 授權。

## 🙏 致謝

由社群用心打造，特別感謝 [Pollen Robotics](https://www.pollen-robotics.com/)、[OpenClaw](https://github.com/openclaw/openclaw) 與 [OpenAI](https://openai.com/) 的技術支持。
