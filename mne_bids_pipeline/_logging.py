"""Logging."""

import datetime
import inspect
import logging
import os

import rich.console
import rich.theme

from .typing import LogKwargsT


class _MBPLogger:
    def __init__(self) -> None:
        self._level = logging.INFO
        self.__console: rich.console.Console | None = None

    # Do lazy instantiation of _console so that pytest's output capture
    # mechanics don't get messed up
    @property
    def _console(self) -> rich.console.Console:
        if isinstance(self.__console, rich.console.Console):
            return self.__console

        force_terminal: bool | None = None
        force_terminal_env = os.getenv("MNE_BIDS_PIPELINE_FORCE_TERMINAL", None)
        if force_terminal_env is not None:
            force_terminal = force_terminal_env.lower() in ("true", "1")
        legacy_windows = None
        legacy_windows_env = os.getenv("MNE_BIDS_PIPELINE_LEGACY_WINDOWS", None)
        if legacy_windows_env is not None:
            legacy_windows = legacy_windows_env.lower() in ("true", "1")
        theme = rich.theme.Theme(
            dict(
                default="white",
                # Rule
                title="bold green",
                # Prefixes
                asctime="green",
                prefix="bold cyan",
                # Messages
                debug="dim",
                info="",
                warning="magenta",
                error="red",
            )
        )
        self.__console = rich.console.Console(
            soft_wrap=True,
            force_terminal=force_terminal,
            legacy_windows=legacy_windows,
            theme=theme,
        )
        return self.__console

    def title(self, title: str) -> None:
        # Align left with ASCTIME offset
        title = f"[title]┌────────┬ {title}[/]"
        self._console.rule(title=title, characters="─", style="title", align="left")

    def end(self, msg: str = "") -> None:
        self._console.print(f"[title]└────────┴ {msg}[/]")

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, level: int) -> None:
        level = int(level)
        self._level = level

    def debug(
        self, msg: str, *, extra: LogKwargsT | dict[str, str] | None = None
    ) -> None:
        self._log_message(kind="debug", msg=msg, **(extra or {}))

    def info(
        self, msg: str, *, extra: LogKwargsT | dict[str, str] | None = None
    ) -> None:
        self._log_message(kind="info", msg=msg, **(extra or {}))

    def warning(
        self, msg: str, *, extra: LogKwargsT | dict[str, str] | None = None
    ) -> None:
        self._log_message(kind="warning", msg=msg, **(extra or {}))

    def error(
        self, msg: str, *, extra: LogKwargsT | dict[str, str] | None = None
    ) -> None:
        self._log_message(kind="error", msg=msg, **(extra or {}))

    def _log_message(
        self,
        kind: str,
        msg: str,
        subject: str | None = None,
        session: str | None = None,
        run: str | None = None,
        emoji: str = "",
    ) -> None:
        this_level = getattr(logging, kind.upper())
        if this_level < self.level:
            return
        # Construct str
        essr = " ".join(x for x in [emoji, subject, session, run] if x)
        if essr:
            essr += " "
        asctime = datetime.datetime.now().strftime("│%H:%M:%S│")
        msg = f"[asctime]{asctime} [/][prefix]{essr}[/][{kind}]{msg}[/]"
        self._console.print(msg)


logger = _MBPLogger()


def gen_log_kwargs(
    message: str,
    *,
    subject: str | int | None = None,
    session: str | int | None = None,
    run: str | int | None = None,
    task: str | None = None,
    emoji: str = "⏳️",
) -> LogKwargsT:
    # Try to figure these out
    assert isinstance(message, str), type(message)
    stack = inspect.stack()
    up_locals = stack[1].frame.f_locals
    if subject is None:
        subject = up_locals.get("subject", None)
    if session is None:
        session = up_locals.get("session", None)
    if run is None:
        run = up_locals.get("run", None)
        if run is None:
            task = task or up_locals.get("task", None)
            if task in ("noise", "rest"):
                run = task

    # Do some nice formatting
    if subject is not None:
        subject = f"sub-{subject}"
    if session is not None:
        session = f"ses-{session}"
    if run is not None:
        run = f"run-{run}"

    # Choose some to be our standards
    emoji = dict(
        cache="✅",
        skip="⏩",
        override="❌",
    ).get(emoji, emoji)
    extra = {"emoji": emoji}
    if subject:
        extra["subject"] = subject
    if session:
        extra["session"] = session
    if run:
        extra["run"] = run

    kwargs: LogKwargsT = {
        "msg": message,
        "extra": extra,
    }
    return kwargs


def _linkfile(uri: str) -> str:
    return f"[link=file://{uri}]{uri}[/link]"


def _is_testing() -> bool:
    return os.getenv("_MNE_BIDS_STUDY_TESTING", "") == "true"
