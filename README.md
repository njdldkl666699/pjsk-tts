# pjsk-tts

Project SEKAI（プロジェクトセカイ）多角色 VITS 语音合成工具，提供 **桌面 GUI** 与 **OpenAI 兼容的 HTTP 服务端** 两种使用方式。

## 功能特性

- **GUI 模式**：图形界面选择配置与模型、多角色切换、语速调节、生成/播放/保存
- **服务端模式**：OpenAI 兼容的 `/v1/audio/speech` 接口，可无缝接入现有 OpenAI SDK / 客户端
- **共享推理引擎**：GUI 与服务端共用 `TtsEngine`，自动根据模型权重匹配符号预设（default / preset2 / preset3 / ipa）
- **灵活配置**：通过 `.env` 或环境变量指定默认配置与模型，也可在 GUI 对话框或 HTTP 请求参数中动态指定
- **多种输出格式**：wav / flac / ogg / mp3 / pcm
- **uv 管理**：单一 `pyproject.toml` 声明全部依赖，Cython 扩展自动编译安装

## 项目结构

```
gui-resource/
├── gui.py                      # GUI 入口
├── server.py                   # 服务端入口
├── pyproject.toml / uv.lock    # 依赖与项目定义
├── setup.py                    # Cython 扩展编译配置
├── .env(.example)              # 环境变量配置
├── configs/                    # 各角色/组合的模型配置（*.json）
├── models/                     # 模型权重（*.pth，不入库）
└── pjsk_tts/
    ├── app.py                  # GUI 实现
    ├── server.py               # FastAPI 服务端
    ├── engine.py               # 共享推理引擎（加载/预设/合成）
    ├── models.py               # VITS SynthesizerTrn
    ├── modules.py / attentions.py / commons.py / transforms.py / utils.py
    ├── text/                   # 文本正则化（cleaners / transform）
    └── monotonic_align/        # 单调对齐搜索（Cython 扩展）
```

## 环境要求

- **Python**：3.13（由 `uv` 自动下载管理）
- **uv**：[安装方式](https://docs.astral.sh/uv/getting-started/installation/)
- **Linux 系统依赖**：
  ```bash
  # PySide6 GUI 需要
  sudo apt install libxcb-cursor0
  # pyopenjtalk 源码编译需要
  sudo apt install cmake build-essential
  ```

## 快速开始

```bash
# 1. 安装依赖（自动创建 .venv、编译 monotonic_align 扩展）
uv sync

# 2. 放置模型文件，例如：
#    models/kanade_v1.0.pth

# 3. 配置默认模型（可选）
cp .env.example .env
#   编辑 PJSK_CONFIG / PJSK_MODEL 指向实际文件

# 4. 启动 GUI
uv run gui.py
```

未配置 `.env` 时，GUI 保持手动选择文件的行为，与原版一致。

## 服务端模式

```bash
uv run server.py
# 或
uv run uvicorn pjsk_tts.server:app --host 0.0.0.0 --port 8000
```

启动时若已配置 `PJSK_CONFIG` / `PJSK_MODEL` 则自动预加载；否则等待请求参数指定。

### 接口

#### `GET /health`

返回加载状态、当前模型、角色列表与环境变量配置。

#### `POST /v1/audio/speech`（OpenAI 兼容）

请求体（JSON）：

| 字段              | 类型    | 说明                                             |
| ----------------- | ------- | ------------------------------------------------ |
| `input`           | string  | 要合成的文本（必填）                             |
| `model`           | string? | 模型显示名，或直接传 `.pth` 文件路径             |
| `voice`           | string? | 角色名或索引（多说话人模型，如 `mmj` 系配置）    |
| `response_format` | string  | `wav`（默认）/ `flac` / `ogg` / `mp3` / `pcm`    |
| `speed`           | float?  | 语速，`>1` 更快（映射 `length_scale = 1/speed`） |
| `config`          | string? | 扩展：配置文件路径（env 未配置时必需）           |
| `model_path`      | string? | 扩展：模型文件路径（env 未配置时必需）           |
| `noise_scale`     | float   | 扩展：默认 `0.667`                               |
| `noise_scale_w`   | float   | 扩展：默认 `0.8`                                 |
| `length_scale`    | float?  | 扩展：时长缩放，优先于 `speed`                   |

响应为音频二进制流，附带 `X-Sample-Rate` 响应头。

**配置优先级**：请求参数 > 环境变量（`PJSK_CONFIG` / `PJSK_MODEL`）。两者皆未提供时返回 `400`。

### 示例

curl：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kanade", "input": "こんにちは、元気ですか？", "speed": 1.2}' \
  -o speech.wav
```

OpenAI SDK（Python）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

with client.audio.speech.with_streaming_response.create(
    model="kanade",  # 已通过 env 预加载时可任意填写
    voice="kanade",
    input="こんにちは、サーバーモードのテストです。",
    response_format="wav",
    speed=1.0,
) as resp:
    resp.stream_to_file("speech.wav")
```

请求参数指定其他模型（无需重启服务，自动热切换）：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "テスト", "config": "configs/mafuyu.json", "model_path": "models/mafuyu.pth", "voice": "white"}' \
  -o speech.wav
```

## 环境变量

| 变量          | 说明                                            | 默认值      |
| ------------- | ----------------------------------------------- | ----------- |
| `PJSK_CONFIG` | 默认配置文件路径（GUI 自动加载 / 服务端预加载） | —           |
| `PJSK_MODEL`  | 默认模型文件路径                                | —           |
| `PJSK_HOST`   | 服务端监听地址                                  | `127.0.0.1` |
| `PJSK_PORT`   | 服务端监听端口                                  | `8000`      |

以上变量均可写入工作目录下的 `.env` 文件（参见 `.env.example`）。

## 开发

```bash
uv sync                       # 安装/同步环境（含 Cython 扩展编译）
uvx ruff format .             # 代码格式化
uvx ruff check .              # 静态检查
uv run gui.py       # 运行 GUI
uv run server.py         # 运行服务端
```

- 日志（loguru）追加写入工作目录 `logs/app.log`，自动轮转（10MB × 5）
- 桌面打包：`pjsk-tts.spec`（PyInstaller）

## 致谢

- [PJSK-MultiGUI](https://github.com/Kanade-nya/PJSK-MultiGUI)（项目介绍，其 README 含模型下载链接）
- [gui-resource](https://github.com/Kanade-nya/gui-resource)（GUI 源码，本项目的前身）
- [VITS](https://github.com/jaywalnut310/vits)（VITS 模型实现）
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)（日语文本正则化）
- 各模型配置来自 PJSK 民间 VITS 训练项目
