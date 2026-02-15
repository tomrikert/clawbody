# 🦞🤖 ClawBody (優化版)

**為您的 OpenClaw AI 代理提供實體機器人身體！**

ClawBody 將 OpenClaw 的 AI 智慧與 Reachy Mini 的表現力機器人身體相結合，利用 OpenAI 的 Realtime API 實現超靈敏的語音對話。您的 OpenClaw 助手 (Clawson) 現在可以在現實世界中看、聽、說與移動。

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

---

## 🌟 最新更新 (過去 24 小時)

- **🎭 動態情感與舞蹈庫**：整合了 Reachy 守護程序 (daemon) 記錄的所有情感與舞蹈，讓機器人更具表現力。
- **👋 自然回合級手勢 (Natural Gestures)**：新增了隨對話自動觸發的自然手勢，讓互動更像人類。
- **🗣️ 語音同步動作**：手勢現在會與語音同步，並加入了身體搖擺 (body_sway) 效果。
- **🎯 關鍵字觸發手勢**：可以根據特定的關鍵字觸發特定的手勢動作。
- **🔗 連線優化**：修復了 OpenClaw 的 CORS 連線問題，並優化了 MediaPipe 追蹤數據格式。

---

## 👁️ 核心功能：人臉追蹤與眼神接觸

**機器人在你說話時會看著你！**

ClawBody 包含即時人臉追蹤功能，使對話感覺自然且引人入勝：

- **自動人臉檢測**：使用 MediaPipe 或 YOLO 以 25Hz 檢測人臉。
- **平滑頭部追蹤**：當您移動時，機器人會平滑地跟隨您的臉。
- **自然眼神接觸**：在對話期間保持參與感。

```bash
# 預設啟動人臉追蹤
clawbody

# 選擇追蹤器（MediaPipe 較輕量，YOLO 更準確）
clawbody --head-tracker mediapipe
```

---

## 🎮 無硬體？沒問題！

**您不需要實體 Reachy Mini 機器人即可使用 ClawBody！**

ClawBody 可與 [Reachy Mini 模擬器](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started) 配合使用。這是一個基於 MuJoCo 的物理模擬，可以在您的電腦上運行。

```bash
# 安裝模擬器支援
pip install "reachy-mini[mujoco]"

# 啟動模擬器 (會開啟 3D 視窗)
# Mac 使用者請用: mjpython -m reachy_mini.daemon.app.main --sim
reachy-mini-daemon --sim

# 在另一個終端機執行 ClawBody
clawbody --gradio
```

---

## 🚀 快速開始

### 1. 安裝環境

```bash
# 複製儲存庫
git clone https://github.com/dAAAb/clawbody
cd clawbody

# 建立並啟動虛擬環境
python -m venv .venv
source .venv/bin/activate

# 安裝 ClawBody 及其依賴
pip install -e ".[mediapipe_vision]"
```

### 2. 配置設定

複製環境變數範本並填入您的金鑰：

```bash
cp .env.example .env
```

編輯 `.env`：
- `OPENAI_API_KEY`: 您的 OpenAI API Key (需支援 Realtime API)
- `OPENCLAW_GATEWAY_URL`: OpenClaw Gateway 的網址 (例如 `http://localhost:18789`)
- `OPENCLAW_TOKEN`: 您的 OpenClaw Token

### 3. 執行

```bash
# 使用模擬器時，建議搭配 --gradio 開啟網頁介面
clawbody --gradio
```

---

## 🛠️ 機器人能力 (Capabilities)

ClawBody 賦予 Clawson 以下實體能力：

| 能力 | 描述 |
|------------|-------------|
| **人臉追蹤** | 對話時自動追蹤並注視人類 |
| **視覺 (See)** | 通過機器人鏡頭捕捉影像並描述環境 |
| **情感 (Emotions)** | 執行預錄的情感動作（如：好奇、快樂、思考） |
| **手勢 (Gestures)** | 與語音同步的自然手勢，增強對話表現力 |
| **舞蹈 (Dance)** | 執行表現力豐富的舞蹈動畫 |

---

## 🤝 貢獻指南

我們歡迎任何形式的貢獻！如果您有新功能的想法或發現 Bug，請提交 Pull Request 或建立 Issue。

- **提交 Issue**：報告錯誤或提出新功能建議。
- **Pull Request**：請確保代碼風格一致並通過測試。

---

## 📄 授權資訊

本專案採用 Apache 2.0 授權 - 詳見 [LICENSE](LICENSE) 檔案。

---

## 🙏 致謝

本專案建立在以下優秀作品之上：
- [Pollen Robotics](https://www.pollen-robotics.com/) (Reachy Mini)
- [OpenClaw](https://github.com/openclaw/openclaw) (AI 助手框架)
- [OpenAI Realtime API](https://openai.com/)
