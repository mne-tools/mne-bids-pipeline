"""Logging."""

import datetime
import inspect
import logging
import os
from typing import Optional, Union

import rich.console
import rich.theme

from .typing import LogKwargsT


class _MBPLogger:
    def __init__(self):
        self._level = logging.INFO

    # Do lazy instantiation of _console so that pytest's output capture
    # mechanics don't get messed up
    @property
    def _console(self):
        try:
            return self.__console
        except AttributeError:
            pass  # need to instantiate it, continue

        force_terminal = os.getenv("MNE_BIDS_PIPELINE_FORCE_TERMINAL", None)
        if force_terminal is not None:
            force_terminal = force_terminal.lower() in ("true", "1")
        legacy_windows = os.getenv("MNE_BIDS_PIPELINE_LEGACY_WINDOWS", None)
        if legacy_windows is not None:
            legacy_windows = legacy_windows.lower() in ("true", "1")
        kwargs = dict(
            soft_wrap=True,
            force_terminal=force_terminal,
            legacy_windows=legacy_windows,
        )
        kwargs["theme"] = rich.theme.Theme(
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
        self.__console = rich.console.Console(**kwargs)
        return self.__console

    def title(self, title):
        # Align left with ASCTIME offset
        title = f"[title]┌────────┬ {title}[/]"
        self._console.rule(title=title, characters="─", style="title", align="left")

    def end(self, msg=""):
        self._console.print(f"[title]└────────┴ {msg}[/]")

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        level = int(level)
        self._level = level

    def debug(self, msg: str, *, extra: Optional[LogKwargsT] = None) -> None:
        self._log_message(kind="debug", msg=msg, **(extra or {}))

    def info(self, msg: str, *, extra: Optional[LogKwargsT] = None) -> None:
        self._log_message(kind="info", msg=msg, **(extra or {}))

    def warning(self, msg: str, *, extra: Optional[LogKwargsT] = None) -> None:
        self._log_message(kind="warning", msg=msg, **(extra or {}))

    def error(self, msg: str, *, extra: Optional[LogKwargsT] = None) -> None:
        self._log_message(kind="error", msg=msg, **(extra or {}))

    def _log_message(
        self,
        kind: str,
        msg: str,
        subject: Optional[Union[str, int]] = None,
        session: Optional[Union[str, int]] = None,
        run: Optional[Union[str, int]] = None,
        emoji: str = "",
    ):
        this_level = getattr(logging, kind.upper())
        if this_level < self.level:
            return
        # Construct str
        essr = [x for x in [emoji, subject, session, run] if x]
        essr = " ".join(essr)
        if essr:
            essr += " "
        asctime = datetime.datetime.now().strftime("│%H:%M:%S│")
        msg = f"[asctime]{asctime} [/][prefix]{essr}[/][{kind}]{msg}[/]"
        self._console.print(msg)


logger = _MBPLogger()


def gen_log_kwargs(
    message: str,
    *,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    task: Optional[str] = None,
    emoji: str = "⏳️",
) -> LogKwargsT:
    # Try to figure these out
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


def _linkfile(uri):
    return f"[link=file://{uri}]{uri}[/link]"


def _is_testing() -> bool:
    return os.getenv("_MNE_BIDS_STUDY_TESTING", "") == "true"
