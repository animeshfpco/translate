import sys
from collections import deque

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

_PENDING = "…"


class OverlayWindow(QObject):
    """Always-on-top dual-pane window (JA transcript / EN translation).

    Call `append_ja(ja)` immediately after ASR — shows JA text and a "…"
    placeholder in the EN pane. Call `update_last_en(en)` when MT finishes
    to replace the placeholder with the real translation.

    Both methods are thread-safe (Qt signal dispatch to the GUI thread).
    """

    _ja_appended = Signal(str)
    _en_updated = Signal(str)

    def __init__(self) -> None:
        self._app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()

        self._window = QMainWindow()
        self._window.setWindowTitle("translate — JA→EN")
        self._window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._window.resize(640, 420)

        central = QWidget()
        layout = QVBoxLayout(central)

        self._ja = QPlainTextEdit()
        self._ja.setReadOnly(True)
        self._ja.setPlaceholderText("Japanese transcript")

        self._en = QPlainTextEdit()
        self._en.setReadOnly(True)
        self._en.setPlaceholderText("English translation")

        layout.addWidget(QLabel("JA"))
        layout.addWidget(self._ja)
        layout.addWidget(QLabel("EN"))
        layout.addWidget(self._en)

        self._window.setCentralWidget(central)
        self._pending_blocks: deque[int] = deque()
        self._ja_appended.connect(self._on_ja_appended)
        self._en_updated.connect(self._on_en_updated)

    def _on_ja_appended(self, ja: str) -> None:
        self._ja.appendPlainText(ja)
        self._en.appendPlainText(_PENDING)
        # appendPlainText fills the initial empty block on the first call and
        # appends a new block thereafter, so blockCount()-1 is the placeholder's index.
        self._pending_blocks.append(self._en.blockCount() - 1)

    def _on_en_updated(self, en: str) -> None:
        if not self._pending_blocks:
            return
        block_num = self._pending_blocks.popleft()
        block = self._en.document().findBlockByNumber(block_num)
        if not block.isValid():
            return
        cursor = QTextCursor(block)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfBlock,
            QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.insertText(en)

    def append_ja(self, ja: str) -> None:
        """Show JA text immediately + a pending placeholder in the EN pane."""
        self._ja_appended.emit(ja)

    def update_last_en(self, en: str) -> None:
        """Replace the pending placeholder with the finished translation."""
        self._en_updated.emit(en)

    def run(self) -> None:
        self._window.show()
        self._app.exec()
