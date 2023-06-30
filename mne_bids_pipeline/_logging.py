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
        kwargs = dict(soft_wrap=True, force_terminal=force_terminal)
        kwargs["theme"] = rich.theme.Theme(
            dict(
                default="white",
                # Prefixes
                asctime="green",
                step="bold cyan",
                # Messages
                debug="dim",
                info="bold",
                warning="bold magenta",
                error="bold red",
            )
        )
        self.__console = rich.console.Console(**kwargs)
        return self.__console

    def rule(self, title="", *, align="center"):
        self.__console.rule(title=title, characters="─", style="rule.line", align=align)

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
        step: Optional[str] = None,
        emoji: str = "",
        box: str = "",
    ):
        this_level = getattr(logging, kind.upper())
        if this_level < self.level:
            return
        if not subject:
            subject = ""
        if not session:
            session = ""
        if not run:
            run = ""
        if not step:
            step = ""
        if step and emoji:
            step = f"{emoji} {step}"
        asctime = datetime.datetime.now().strftime("[%H:%M:%S]")
        msg = (
            f"[asctime]{asctime}[/asctime] "
            f"[step]{box}{step}{subject}{session}{run}[/step]"
            f"[{kind}]{msg}[/{kind}]"
        )
        self._console.print(msg)


logger = _MBPLogger()


def gen_log_kwargs(
    message: str,
    *,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    task: Optional[str] = None,
    step: Optional[str] = None,
    emoji: str = "⏳️",
    box: str = "│ ",
) -> LogKwargsT:
    from ._run import _get_step_path, _short_step_path

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
    if step is None:
        step_path = _get_step_path(stack)
        if step_path:
            step = _short_step_path(_get_step_path())
        else:
            step = ""

    # Do some nice formatting
    if subject is not None:
        subject = f" sub-{subject}"
    if session is not None:
        session = f" ses-{session}"
    if run is not None:
        run = f" run-{run}"
    if step != "":
        # need an extra space
        message = f" {message}"

    # Choose some to be our standards
    emoji = dict(
        cache="✅",
        skip="⏩",
        override="❌",
    ).get(emoji, emoji)
    extra = {
        "step": f"{emoji} {step}",
        "box": box,
    }
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
