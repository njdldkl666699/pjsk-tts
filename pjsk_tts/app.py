import os
import random
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from loguru import logger
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .engine import DEFAULT_SAMPLE_RATE, ENV_CONFIG, ENV_MODEL, TtsEngine

# 日志追加写入工作目录 logs/ 下（loguru 默认仍保留 stderr 彩色输出）
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "app.log",
    rotation="10 MB",
    retention=5,
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}",
)


class Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pjsk-tts")
        QApplication.setFont(QFont("SimSun", 12))
        self.resize(1080, 720)
        self._center_on_screen()

        self.current_audio = None
        self.current_audio_path: Path | None = None
        self.current_sr = DEFAULT_SAMPLE_RATE
        self.engine = TtsEngine(on_log=self._ui_log)

        self._init_ui()
        self._apply_env_defaults()

    def _init_ui(self) -> None:
        layout = QGridLayout(self)

        # --- Model selection group
        model_group = QGroupBox("选择模型")
        mg_layout = QVBoxLayout()
        self.cfg_path_edit = QLineEdit()
        self.cfg_path_edit.setReadOnly(True)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)

        cfg_btn = QPushButton("选择配置")
        cfg_btn.clicked.connect(self._select_config)
        model_btn = QPushButton("选择文件")
        model_btn.clicked.connect(self._select_model)

        row1 = QHBoxLayout()
        row1.addWidget(cfg_btn)
        row1.addWidget(self.cfg_path_edit)
        row2 = QHBoxLayout()
        row2.addWidget(model_btn)
        row2.addWidget(self.model_path_edit)
        mg_layout.addLayout(row1)
        mg_layout.addLayout(row2)
        model_group.setLayout(mg_layout)
        layout.addWidget(model_group, 0, 0)

        # --- Speaker selection
        self.spk_group = QGroupBox("当前角色")
        spk_layout = QVBoxLayout()
        self.spk_combo = QComboBox()
        self.spk_combo.currentTextChanged.connect(self._speaker_changed)
        spk_btn = QPushButton("确定")
        spk_btn.clicked.connect(self._confirm_speaker)
        self.spk_label = QLabel("当前选择：")
        spk_bottom = QHBoxLayout()
        spk_bottom.addWidget(self.spk_label)
        spk_bottom.addStretch()
        spk_bottom.addWidget(spk_btn)
        spk_layout.addWidget(self.spk_combo)
        spk_layout.addLayout(spk_bottom)
        self.spk_group.setLayout(spk_layout)
        layout.addWidget(self.spk_group, 1, 0)

        # --- TTS input
        tts_group = QGroupBox("语音合成")
        tts_layout = QVBoxLayout()
        tts_layout.addWidget(QLabel("输入日语原文"))
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(150)
        tts_layout.addWidget(self.text_edit)

        slider_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"当前语速：{v / 100:.2f}")
        )
        self.speed_label = QLabel("当前语速：1.00")
        gen_btn = QPushButton("生成")
        gen_btn.clicked.connect(self._generate_audio)
        slider_row.addWidget(self.speed_slider)
        slider_row.addWidget(self.speed_label)
        slider_row.addWidget(gen_btn)
        tts_layout.addLayout(slider_row)
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group, 2, 0, 2, 1)

        # --- Output section
        out_group = QGroupBox("输出")
        out_layout = QHBoxLayout()
        play_btn = QPushButton("播放")
        play_btn.clicked.connect(self._play_audio)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._save_audio)
        out_layout.addWidget(play_btn)
        out_layout.addWidget(save_btn)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group, 4, 0)

        # --- Log / info panel
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 0, 1, 5, 1)

    def _select_config(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置", "./", "Config (*.json)")
        if not file_path:
            return
        self.cfg_path_edit.setText(file_path)
        self._load_config(Path(file_path))

    def _select_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "./", "Model (*.pth)")
        if not file_path:
            return
        self.model_path_edit.setText(file_path)
        self._load_model(Path(file_path))

    def _load_config(self, cfg_path: Path) -> None:
        try:
            self.engine.load_config(cfg_path)
            self._populate_speaker_combo()
            self._log(f"配置文件加载成功: {cfg_path.name}")
        except Exception as exc:
            self._error(f"配置文件加载失败: {exc}")

    def _load_model(self, model_path: Path) -> None:
        try:
            self.engine.load_model(model_path)
        except Exception as exc:
            self._error(f"模型文件加载失败: {exc}")

    def _apply_env_defaults(self) -> None:
        """若通过 .env / 环境变量配置了路径，则启动时自动加载。"""
        cfg = os.getenv(ENV_CONFIG)
        mdl = os.getenv(ENV_MODEL)
        if not cfg or not mdl:
            return
        cfg_path, model_path = Path(cfg), Path(mdl)
        if not cfg_path.is_file() or not model_path.is_file():
            self._log(f"环境变量指向的文件不存在，跳过默认加载: {cfg} / {mdl}")
            return
        self.cfg_path_edit.setText(str(cfg_path))
        self.model_path_edit.setText(str(model_path))
        self._load_config(cfg_path)
        self._load_model(model_path)

    def _populate_speaker_combo(self) -> None:
        self.spk_combo.clear()
        self.spk_combo.addItems(self.engine.speakers)
        # Trigger label update
        self._speaker_changed(self.spk_combo.currentText())

    def _speaker_changed(self, name: str) -> None:
        if self.engine.multi_speaker:
            self.engine.speaker_id = self.spk_combo.currentIndex()
        self.spk_label.setText(f"当前选择：{name}")

    def _confirm_speaker(self) -> None:
        self._log(self.spk_label.text())

    def _center_on_screen(self) -> None:
        geometry = self.frameGeometry()
        geometry.moveCenter(self.screen().availableGeometry().center())
        self.move(geometry.topLeft())

    def _play_audio(self):
        if not self.current_audio_path or not self.current_audio_path.exists():
            self._error("没有可播放的音频文件。请先生成音频。")
            return
        try:
            effect = QSoundEffect(self)
            effect.setSource(QUrl.fromLocalFile(str(self.current_audio_path)))
            effect.setLoopCount(1)
            effect.setVolume(1.0)
            self._playing_effect = effect
            effect.play()
            self._log(f"播放: {self.current_audio_path.name}")
        except Exception as e:
            self._error(f"播放失败: {e}")

    def _generate_audio(self) -> None:
        if not self.engine.loaded:
            self._error("模型或配置未加载。请先选择模型和配置文件。")
            return
        raw_text = self.text_edit.toPlainText().strip()
        if not raw_text:
            self._error("请输入要合成的文本。")
            return
        self._log("开始生成音频...")

        try:
            audio, sr = self.engine.synthesize(
                raw_text, length_scale=self.speed_slider.value() / 100.0
            )
            self.current_audio = audio
            self.current_sr = sr
            # Save to temp file
            temp_dir = Path(tempfile.gettempdir()) / "pjsk-tts"
            temp_dir.mkdir(exist_ok=True)
            safe_name = raw_text.replace("?", "").strip()[:10] or "voice"
            self.current_audio_path = temp_dir / f"{safe_name}_{random.randint(1000, 9999)}.wav"
            sf.write(self.current_audio_path, audio, sr)
            self._log("音频生成成功。")
        except Exception as exc:
            self._error(f"推理失败: {exc}")

    def _save_audio(self) -> None:
        if self.current_audio is None:
            self._error("没有生成音频可保存。请先生成音频。")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save WAV", "result.wav", "WAV (*.wav)")
        if not target:
            return
        try:
            sf.write(target, self.current_audio, self.current_sr)
            self._log(f"保存到: {target}")
        except Exception as exc:
            self._error(f"保存失败: {exc}")

    def _ui_log(self, message: str) -> None:
        """engine 回调：仅写入界面日志面板（loguru 已在 engine 内记录）。"""
        self.log_view.append(message)

    def _log(self, message: str) -> None:
        logger.info(message)
        self.log_view.append(message)

    def _error(self, message: str) -> None:
        logger.error(message)
        self.log_view.append(f"Error: {message}")


def main() -> None:
    try:
        app = QApplication(sys.argv)
        ex = Window()
        ex.show()
        sys.exit(app.exec())
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}")


if __name__ == "__main__":
    main()
