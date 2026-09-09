"""共享 TTS 推理引擎：配置/模型加载、符号预设、文本清洗与语音合成。

供 GUI（app.py）与服务端（server.py）共同使用。
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from . import commons, utils
from .models import SynthesizerTrn
from .text import transform
from .text.cleaners import japanese_cleaners, japanese_cleaners2, japanese_tokenization_cleaners

#: 环境变量名：配置文件路径 / 模型文件路径（亦可通过 .env 配置）
ENV_CONFIG = "PJSK_CONFIG"
ENV_MODEL = "PJSK_MODEL"

DEFAULT_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class SymbolPreset:
    id: int
    symbols: list[str]


SYMBOL_PRESETS: dict[str, SymbolPreset] = {
    "default": SymbolPreset(1, list(' !"&*,-.?ABCINU[]abcdefghijklmnoprstuwyz{}~')),
    "preset2": SymbolPreset(2, ["_", *list(",.!?-"), *list("AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ ")]),
    "preset3": SymbolPreset(
        3, ["_", *list(",.!?-~…"), *list("AEINOQUabdefghijkmnoprstuvwyzʃʦ↓↑ ")]
    ),
    "ipa": SymbolPreset(
        4,
        [
            "_",
            *list(';:,.!?¡¿—…"«»“” '),
            *list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
            *list(
                "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            ),
        ],
    ),
}

CONFIG_TO_PRESET = {
    "mmj.json": "default",
    "vbs.json": "default",
    "ws.json": "default",
    "mafuyu.json": "default",
}

MULTI_SPK_GROUPS = {
    "mmj.json": ["minori", "haruka", "airi", "shizuku"],
    "vbs.json": ["akito", "an", "kohane", "toya"],
    "ws.json": ["emu", "nene", "rui", "tsukasa"],
    "mafuyu.json": ["white", "black"],
}

_CLEANER_MAP = {
    1: japanese_tokenization_cleaners,
    2: japanese_cleaners,
    3: japanese_cleaners2,
    4: japanese_tokenization_cleaners,
}


def clean_text(text: str, preset: SymbolPreset) -> torch.Tensor:
    """Convert raw text to tensor according to preset symbols."""
    cleaner = _CLEANER_MAP.get(preset.id, japanese_tokenization_cleaners)
    seq = transform.cleaned_text_to_sequence(cleaner(text), preset.symbols)
    return torch.LongTensor(commons.intersperse(seq, 0))


class TtsEngine:
    """封装配置/模型加载与语音合成，GUI 与服务端共用。"""

    def __init__(self, on_log: Callable[[str], None] | None = None) -> None:
        self._on_log = on_log
        self.hps = None
        self.model: SynthesizerTrn | None = None
        self.preset = SYMBOL_PRESETS["default"]
        self.multi_speaker = False
        self.speaker_id = 0
        self.speakers: list[str] = []
        self.config_path: Path | None = None
        self.model_path: Path | None = None

    # ------------------------------------------------------------- 状态属性
    def _log(self, message: str) -> None:
        logger.info(message)
        if self._on_log is not None:
            self._on_log(message)

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.hps is not None

    @property
    def sample_rate(self) -> int:
        if self.hps is None:
            return DEFAULT_SAMPLE_RATE
        return int(getattr(self.hps.data, "sampling_rate", DEFAULT_SAMPLE_RATE))

    # ------------------------------------------------------------- 加载流程
    def load_config(self, cfg_path: str | Path) -> list[str]:
        """加载配置文件，返回角色名列表。"""
        cfg_path = Path(cfg_path)
        self.hps = utils.get_hparams_from_file(str(cfg_path))
        self.config_path = cfg_path
        preset_key = CONFIG_TO_PRESET.get(cfg_path.name, "default")
        self.preset = SYMBOL_PRESETS[preset_key]
        self._log(f"使用符号预设: {preset_key}")

        speakers = MULTI_SPK_GROUPS.get(cfg_path.name)
        if speakers:
            self.speakers = list(speakers)
            self.multi_speaker = True
        else:
            self.speakers = [cfg_path.stem]
            self.multi_speaker = False
        self.speaker_id = 0
        return self.speakers

    def _match_preset_to_checkpoint(self, model_path: Path) -> None:
        """根据 checkpoint 中符号嵌入矩阵的大小自动选择匹配的符号预设。"""
        checkpoint = torch.load(str(model_path), map_location="cpu")
        n_symbols = checkpoint["model"]["enc_p.emb.weight"].shape[0]
        if n_symbols == len(self.preset.symbols):
            return
        for key, preset in SYMBOL_PRESETS.items():
            if len(preset.symbols) == n_symbols:
                self.preset = preset
                self._log(
                    f"符号数与当前预设不符，已自动切换符号预设: {key}（模型符号数 {n_symbols}）"
                )
                return
        self._log(f"警告: 未找到符号数为 {n_symbols} 的预设，保持当前预设。")

    def load_model(self, model_path: str | Path) -> None:
        """加载模型权重（需先加载配置）。"""
        if self.hps is None:
            raise RuntimeError("请先载入配置文件。")
        model_path = Path(model_path)
        self._match_preset_to_checkpoint(model_path)
        self.model = SynthesizerTrn(
            len(self.preset.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.model.eval()
        utils.load_checkpoint(str(model_path), self.model, None)
        self.model_path = model_path
        self._log(f"模型文件加载成功: {model_path.name}")

    def load(self, cfg_path: str | Path, model_path: str | Path) -> None:
        """依次加载配置与模型。"""
        self.load_config(cfg_path)
        self.load_model(model_path)

    def matches(self, cfg_path: str | Path, model_path: str | Path) -> bool:
        """判断当前已加载的组合是否与给定路径一致（路径按 realpath 归一化）。"""
        if self.config_path is None or self.model_path is None:
            return False
        return os.path.realpath(self.config_path) == os.path.realpath(
            str(cfg_path)
        ) and os.path.realpath(self.model_path) == os.path.realpath(str(model_path))

    # ------------------------------------------------------------- 推理
    def resolve_speaker(self, voice: str | int | None) -> int:
        """将角色（名字或索引）解析为 speaker_id；单说话人模型恒为 0。"""
        if not self.multi_speaker:
            return 0
        if voice is None:
            return self.speaker_id
        name = str(voice)
        if name.isdigit() and int(name) < len(self.speakers):
            return int(name)
        if name in self.speakers:
            return self.speakers.index(name)
        raise ValueError(f"未知角色 {voice!r}，可用角色: {', '.join(self.speakers)}")

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | int | None = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """合成语音，返回 (float32 音频数组, 采样率)。"""
        if not self.loaded:
            raise RuntimeError("模型或配置未加载。")
        speaker_id = self.resolve_speaker(speaker)
        text = text.replace("\n", " ").strip()
        if not text:
            raise ValueError("合成文本为空。")

        stn = clean_text(text, self.preset)
        x_tst = stn.unsqueeze(0)
        x_len = torch.LongTensor([stn.size(0)])
        infer_kwargs = {
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "length_scale": length_scale,
        }
        with torch.no_grad():
            if self.multi_speaker:
                audio = (
                    self.model.infer(
                        x_tst, x_len, sid=torch.LongTensor([speaker_id]), **infer_kwargs
                    )[0][0, 0]
                    .cpu()
                    .numpy()
                )
            else:
                audio = self.model.infer(x_tst, x_len, **infer_kwargs)[0][0, 0].cpu().numpy()
        return audio, self.sample_rate
