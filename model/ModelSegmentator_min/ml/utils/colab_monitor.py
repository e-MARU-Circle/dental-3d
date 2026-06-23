"""ColabMonitor — Google Colab学習監視ヘルパー (PyTorch, 1ファイル完結).

機能:
  - Discord Webhook通知 (学習開始/進捗/完了/エラー/切断, JST表示)
  - チェックポイント自動保存 (Nエポックごと, 最新3世代保持)
  - 自動再開 (起動時に最新チェックポイント検出)
  - GPU監視 (バックグラウンドで定期通知)
  - エラー時緊急保存 + Discord通知

使い方:
    monitor = ColabMonitor(
        webhook_url="https://discord.com/api/webhooks/...",
        project_name="stage1_v6",
        checkpoint_dir="/content/drive/MyDrive/ckpts",
        checkpoint_every=5,
        notify_every=10,
    )
    monitor.on_train_start(model, optimizer, epochs=100)
    for epoch in range(monitor.start_epoch, 100):
        loss = train(model)
        acc = evaluate(model)
        monitor.on_epoch_end(epoch, model, optimizer, loss=loss, accuracy=acc)
    monitor.on_train_complete()
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import torch

JST = timezone(timedelta(hours=9))


def _now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def _send_discord(webhook_url: str, content: str, username: str = "ColabMonitor") -> None:
    """Discord Webhookにメッセージを送信 (標準ライブラリのみ)."""
    if not webhook_url:
        return
    payload = json.dumps({"username": username, "content": content}).encode("utf-8")
    req = Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urlopen(req, timeout=10)
    except (URLError, OSError):
        pass  # 通知失敗は無視して学習を続行


def _gpu_stats() -> Dict[str, str]:
    """nvidia-smi からGPU情報を取得."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "gpu_util": f"{parts[1]}%",
                "mem_used": f"{parts[2]} MiB",
                "mem_total": f"{parts[3]} MiB",
                "mem_pct": f"{100 * int(parts[2]) / max(1, int(parts[3])):.0f}%",
                "temp": f"{parts[4]}°C",
            }
    except Exception:
        pass
    return {}


class ColabMonitor:
    """Google Colab学習監視クラス."""

    def __init__(
        self,
        webhook_url: str = "",
        project_name: str = "model",
        checkpoint_dir: str = "/content/drive/MyDrive/checkpoints",
        checkpoint_every: int = 5,
        notify_every: int = 10,
        gpu_watch_interval: int = 300,
        max_keep: int = 3,
        extra_save: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            webhook_url: Discord Webhook URL (空文字で通知無効).
            project_name: プロジェクト名 (チェックポイントファイル名に使用).
            checkpoint_dir: チェックポイント保存先 (Google Drive推奨).
            checkpoint_every: Nエポックごとにチェックポイント保存.
            notify_every: Nエポックごとに進捗通知.
            gpu_watch_interval: GPU監視通知の間隔 (秒). 0で無効.
            max_keep: 保持するチェックポイント世代数.
            extra_save: チェックポイントに追加保存する辞書 (config等).
        """
        self.webhook_url = webhook_url
        self.project_name = project_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.notify_every = notify_every
        self.gpu_watch_interval = gpu_watch_interval
        self.max_keep = max_keep
        self.extra_save = extra_save or {}

        self.start_epoch: int = 0
        self._total_epochs: int = 0
        self._train_start_time: float = 0.0
        self._epoch_times: list[float] = []
        self._best_loss: float = float("inf")
        self._model_ref: Optional[torch.nn.Module] = None
        self._optim_ref: Optional[torch.optim.Optimizer] = None
        self._gpu_thread: Optional[threading.Thread] = None
        self._gpu_stop = threading.Event()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 最新チェックポイントから再開情報を検出
        self._detect_resume()

        # エラー時緊急保存を登録
        self._register_crash_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_train_start(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        scheduler: Any = None,
    ) -> None:
        """学習開始時に呼ぶ. チェックポイントがあれば自動ロード."""
        self._model_ref = model
        self._optim_ref = optimizer
        self._total_epochs = epochs
        self._train_start_time = time.time()

        # Resume: load weights
        if self.start_epoch > 0:
            ckpt_path = self._latest_checkpoint_path()
            if ckpt_path and ckpt_path.exists():
                self._load_checkpoint(ckpt_path, model, optimizer, scheduler)

        n_params = sum(p.numel() for p in model.parameters())
        gpu = _gpu_stats()
        gpu_info = f"{gpu.get('name', 'N/A')} ({gpu.get('mem_total', '?')})" if gpu else "N/A"

        msg = (
            f"**{self.project_name}** 学習開始\n"
            f"```\n"
            f"Time     : {_now_jst()}\n"
            f"GPU      : {gpu_info}\n"
            f"Params   : {n_params:,}\n"
            f"Epochs   : {self.start_epoch} -> {epochs}\n"
            f"Ckpt dir : {self.checkpoint_dir}\n"
            f"Save every {self.checkpoint_every} ep, keep {self.max_keep}\n"
            f"```"
        )
        self._notify(msg)
        self._start_gpu_watch()

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        **metrics: float,
    ) -> None:
        """エポック終了時に呼ぶ. 自動保存+通知を判定."""
        self._model_ref = model
        self._optim_ref = optimizer
        self._epoch_times.append(time.time())

        loss = metrics.get("loss") or metrics.get("val_loss")
        is_best = False
        if loss is not None and loss < self._best_loss:
            self._best_loss = loss
            is_best = True

        # チェックポイント保存
        should_save = (
            (epoch + 1) % self.checkpoint_every == 0
            or is_best
            or (epoch + 1) == self._total_epochs
        )
        if should_save:
            self._save_checkpoint(epoch, model, optimizer, scheduler, metrics, is_best)

        # 進捗通知
        should_notify = (
            (epoch + 1) % self.notify_every == 0
            or is_best
            or (epoch + 1) == self._total_epochs
        )
        if should_notify:
            self._notify_progress(epoch, metrics, is_best)

    def on_train_complete(self) -> None:
        """学習完了時に呼ぶ."""
        self._stop_gpu_watch()
        elapsed = time.time() - self._train_start_time
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        msg = (
            f"**{self.project_name}** 学習完了\n"
            f"```\n"
            f"Time    : {_now_jst()}\n"
            f"Elapsed : {h}h {m}m\n"
            f"Best loss: {self._best_loss:.4f}\n"
            f"```"
        )
        self._notify(msg)

    def on_error(self, error: Exception) -> None:
        """手動エラー報告用."""
        self._emergency_save(error)

    # ------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self._best_loss,
            "project_name": self.project_name,
            "timestamp": _now_jst(),
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
        save_dict.update(self.extra_save)

        # 世代管理: project_ep{epoch:04d}.pth
        ckpt_name = f"{self.project_name}_ep{epoch:04d}.pth"
        ckpt_path = self.checkpoint_dir / ckpt_name
        try:
            torch.save(save_dict, ckpt_path)
        except Exception as e:
            self._notify(f"**{self.project_name}** ckpt保存失敗: {e}")
            return

        # Best を別名でコピー
        if is_best:
            best_path = self.checkpoint_dir / f"{self.project_name}_best.pth"
            try:
                torch.save(save_dict, best_path)
            except Exception:
                pass

        # 古い世代を削除 (最新 max_keep 個 + best を保持)
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        pattern = str(self.checkpoint_dir / f"{self.project_name}_ep*.pth")
        ckpts = sorted(glob.glob(pattern))
        if len(ckpts) <= self.max_keep:
            return
        for old in ckpts[: len(ckpts) - self.max_keep]:
            try:
                os.remove(old)
            except OSError:
                pass

    def _detect_resume(self) -> None:
        """最新チェックポイントを検出して start_epoch を設定."""
        path = self._latest_checkpoint_path()
        if path and path.exists():
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self.start_epoch = ckpt.get("epoch", -1) + 1
                self._best_loss = ckpt.get("best_loss", float("inf"))
            except Exception:
                self.start_epoch = 0

    def _latest_checkpoint_path(self) -> Optional[Path]:
        pattern = str(self.checkpoint_dir / f"{self.project_name}_ep*.pth")
        ckpts = sorted(glob.glob(pattern))
        if ckpts:
            return Path(ckpts[-1])
        return None

    def _load_checkpoint(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass  # optimizer構造変更時はスキップ
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        self._best_loss = ckpt.get("best_loss", float("inf"))
        self._notify(
            f"**{self.project_name}** チェックポイント復元: `{path.name}` (epoch {ckpt.get('epoch', '?')})"
        )

    # ------------------------------------------------------------------
    # Crash / Emergency Save
    # ------------------------------------------------------------------

    def _register_crash_handlers(self) -> None:
        self._orig_excepthook = sys.excepthook
        sys.excepthook = self._crash_excepthook
        atexit.register(self._atexit_handler)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass  # Colab worker threadではsignal登録不可の場合あり

    def _crash_excepthook(self, exc_type, exc_value, exc_tb):
        self._emergency_save(exc_value)
        if self._orig_excepthook:
            self._orig_excepthook(exc_type, exc_value, exc_tb)

    def _signal_handler(self, signum, frame):
        self._emergency_save(RuntimeError(f"Signal {signum} received"))
        raise SystemExit(1)

    def _atexit_handler(self):
        self._stop_gpu_watch()

    def _emergency_save(self, error: Optional[Exception] = None) -> None:
        """緊急保存: 現在のmodel/optimizerをDriveに保存 + Discord通知."""
        tb_str = traceback.format_exc() if error else ""
        error_msg = str(error) if error else "Unknown"

        saved_path = "N/A"
        if self._model_ref is not None:
            try:
                emergency_name = f"{self.project_name}_emergency.pth"
                emergency_path = self.checkpoint_dir / emergency_name
                save_dict = {
                    "model_state_dict": self._model_ref.state_dict(),
                    "best_loss": self._best_loss,
                    "project_name": self.project_name,
                    "timestamp": _now_jst(),
                    "error": error_msg,
                }
                if self._optim_ref is not None:
                    save_dict["optimizer_state_dict"] = self._optim_ref.state_dict()
                torch.save(save_dict, emergency_path)
                saved_path = str(emergency_path)
            except Exception:
                saved_path = "SAVE FAILED"

        # エラーメッセージを短縮
        tb_short = tb_str[-500:] if len(tb_str) > 500 else tb_str

        msg = (
            f"**{self.project_name}** エラー発生\n"
            f"```\n"
            f"Time : {_now_jst()}\n"
            f"Error: {error_msg}\n"
            f"Saved: {saved_path}\n"
            f"```\n"
        )
        if tb_short:
            msg += f"```python\n{tb_short}\n```"
        self._notify(msg)

    # ------------------------------------------------------------------
    # GPU Watch (Background Thread)
    # ------------------------------------------------------------------

    def _start_gpu_watch(self) -> None:
        if self.gpu_watch_interval <= 0 or not self.webhook_url:
            return
        self._gpu_stop.clear()
        self._gpu_thread = threading.Thread(target=self._gpu_watch_loop, daemon=True)
        self._gpu_thread.start()

    def _stop_gpu_watch(self) -> None:
        self._gpu_stop.set()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=5)
            self._gpu_thread = None

    def _gpu_watch_loop(self) -> None:
        while not self._gpu_stop.wait(timeout=self.gpu_watch_interval):
            gpu = _gpu_stats()
            if not gpu:
                continue
            msg = (
                f"`[GPU] {_now_jst()}` "
                f"{gpu['name']} | "
                f"Util {gpu['gpu_util']} | "
                f"Mem {gpu['mem_used']}/{gpu['mem_total']} ({gpu['mem_pct']}) | "
                f"Temp {gpu['temp']}"
            )
            self._notify(msg)

    # ------------------------------------------------------------------
    # Discord Notification
    # ------------------------------------------------------------------

    def _notify(self, content: str) -> None:
        _send_discord(self.webhook_url, content, username=f"Colab: {self.project_name}")

    def _notify_progress(self, epoch: int, metrics: Dict[str, float], is_best: bool) -> None:
        elapsed = time.time() - self._train_start_time
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        pct = (epoch + 1) / max(1, self._total_epochs) * 100

        # ETA
        if len(self._epoch_times) >= 2:
            avg_sec = (self._epoch_times[-1] - self._epoch_times[0]) / max(1, len(self._epoch_times) - 1)
            remaining = (self._total_epochs - epoch - 1) * avg_sec
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)
            eta_str = f"{eta_h}h {eta_m}m"
        else:
            eta_str = "calculating..."

        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        best_marker = " **[BEST]**" if is_best else ""

        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        msg = (
            f"**{self.project_name}** Epoch {epoch + 1}/{self._total_epochs}{best_marker}\n"
            f"`{bar}` {pct:.0f}%\n"
            f"```\n"
            f"Metrics : {metrics_str}\n"
            f"Elapsed : {h}h {m}m | ETA: {eta_str}\n"
            f"Best    : {self._best_loss:.4f}\n"
            f"```"
        )
        self._notify(msg)
