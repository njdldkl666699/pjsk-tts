"""FastAPI 服务端模式：提供 OpenAI 兼容的 /v1/audio/speech 接口。

配置来源优先级：请求参数 > 环境变量（PJSK_CONFIG / PJSK_MODEL，支持 .env）。
两者均未提供时返回 400，提示需要配置。

启动：uv run server.py（或 uv run uvicorn pjsk_tts.server:app）
监听：PJSK_HOST（默认 127.0.0.1）/ PJSK_PORT（默认 8000）
"""

import io
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

from .engine import ENV_CONFIG, ENV_MODEL, TtsEngine

#: 支持的音频格式: 名称 -> (media_type, soundfile 格式)
AUDIO_FORMATS: dict[str, tuple[str, str | None]] = {
    "wav": ("audio/wav", "WAV"),
    "flac": ("audio/flac", "FLAC"),
    "ogg": ("audio/ogg", "OGG"),
    "mp3": ("audio/mpeg", "MP3"),
    "pcm": ("application/octet-stream", None),  # 原始 int16 小端 PCM
}

engine = TtsEngine()
_lock = threading.Lock()


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 兼容请求体（含扩展字段）。"""

    model: str | None = Field(None, description="模型显示名或 .pth 文件路径")
    input: str = Field(description="要合成的文本")
    voice: str | None = Field(None, description="角色名或索引（多说话人模型）")
    response_format: str = Field("wav", description="wav/flac/ogg/mp3/pcm")
    speed: float | None = Field(
        None, gt=0, description="语速，>1 更快（映射 length_scale=1/speed）"
    )
    # ---- 扩展字段（非 OpenAI 标准）----
    config: str | None = Field(None, description="配置文件路径（env 未配置时必需）")
    model_path: str | None = Field(None, description="模型文件路径（env 未配置时必需）")
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    length_scale: float | None = Field(None, description="时长缩放，与 speed 二选一，优先于 speed")


def _resolve_paths(req: SpeechRequest) -> tuple[str, str]:
    """按 请求参数 > 环境变量 的优先级解析 (config, model)。"""
    model_path = req.model_path
    if model_path is None and req.model and (req.model.endswith(".pth") or os.sep in req.model):
        model_path = req.model
    cfg = req.config or os.getenv(ENV_CONFIG)
    mdl = model_path or os.getenv(ENV_MODEL)
    if not cfg or not mdl:
        raise HTTPException(
            status_code=400,
            detail=(
                "未指定 config/model：请配置 PJSK_CONFIG 与 PJSK_MODEL 环境变量（或 .env），"
                "或在请求中传递 config 与 model_path（或以 .pth 路径作为 model）字段。"
            ),
        )
    if not os.path.isfile(cfg):
        raise HTTPException(status_code=400, detail=f"配置文件不存在: {cfg}")
    if not os.path.isfile(mdl):
        raise HTTPException(status_code=400, detail=f"模型文件不存在: {mdl}")
    return cfg, mdl


def _ensure_loaded(req: SpeechRequest) -> None:
    cfg, mdl = _resolve_paths(req)
    if engine.matches(cfg, mdl):
        return
    try:
        engine.load(cfg, mdl)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {exc}") from exc


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """启动时若环境变量已配置则预加载模型。"""
    cfg, mdl = os.getenv(ENV_CONFIG), os.getenv(ENV_MODEL)
    if cfg and mdl:
        try:
            engine.load(cfg, mdl)
            logger.info("已从环境变量预加载: config={}, model={}", cfg, mdl)
        except Exception as exc:
            logger.error("预加载失败（等待请求时按参数加载）: {}", exc)
    else:
        logger.info("未配置 {} / {}，等待请求参数指定模型", ENV_CONFIG, ENV_MODEL)
    yield


app = FastAPI(title="pjsk-tts Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded": engine.loaded,
        "config": str(engine.config_path) if engine.config_path else None,
        "model": str(engine.model_path) if engine.model_path else None,
        "speakers": engine.speakers,
        "multi_speaker": engine.multi_speaker,
        "env": {"config": os.getenv(ENV_CONFIG), "model": os.getenv(ENV_MODEL)},
    }


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest) -> Response:
    """OpenAI 兼容的语音合成接口。"""
    fmt = req.response_format.lower()
    if fmt not in AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 response_format: {fmt}，支持: {', '.join(AUDIO_FORMATS)}",
        )

    with _lock:
        _ensure_loaded(req)
        if req.length_scale is not None:
            length_scale = req.length_scale
        elif req.speed is not None:
            length_scale = 1.0 / req.speed
        else:
            length_scale = 1.0
        try:
            audio, sr = engine.synthesize(
                req.input,
                speaker=req.voice,
                noise_scale=req.noise_scale,
                noise_scale_w=req.noise_scale_w,
                length_scale=length_scale,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    media_type, sf_format = AUDIO_FORMATS[fmt]
    if sf_format is None:  # pcm
        data = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    else:
        buf = io.BytesIO()
        try:
            sf.write(buf, audio, sr, format=sf_format)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"音频编码失败({fmt}): {exc}") from exc
        data = buf.getvalue()

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "X-Sample-Rate": str(sr),
            "Content-Disposition": f'attachment; filename="speech.{fmt}"',
        },
    )


def main() -> None:
    host = os.getenv("PJSK_HOST", "127.0.0.1")
    port = int(os.getenv("PJSK_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
