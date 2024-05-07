"""Custom hooks for MkDocs-Material."""

import logging
from typing import Any

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

from mne_bids_pipeline._docs import _ParseConfigSteps

logger = logging.getLogger("mkdocs")

config_updated = False


# This hack can be cleaned up once this is resolved:
# https://github.com/mkdocstrings/mkdocstrings/issues/615#issuecomment-1971568301
def on_pre_build(config: MkDocsConfig) -> None:
    """Monkey patch mkdocstrings-python jinja template to have global vars."""
    python_handler = config["plugins"]["mkdocstrings"].get_handler("python")
    python_handler.env.globals["pipeline_steps"] = _ParseConfigSteps()


# Ideally there would be a better hook, but it's unclear if context can
# be obtained any earlier
def on_template_context(
    context: dict[str, Any],
    template_name: str,
    config: MkDocsConfig,
) -> None:
    """Update the copyright in the footer."""
    global config_updated
    if not config_updated:
        config_updated = True
        now = context["build_date_utc"].strftime("%Y/%m/%d")
        config.copyright = f"{config.copyright}, last updated {now}"
        logger.info(f"Updated copyright to {config.copyright}")


_EMOJI_MAP = {
    "🏆": ":trophy:",
    "🛠️": ":tools:",
    "📘": ":blue_book:",
    "🧑‍🤝‍🧑": ":people_holding_hands_tone1:",
    "💻": ":computer:",
    "🆘": ":sos:",
    "👣": ":footprints:",
    "⏩": ":fast_forward:",
    "⏏️": ":eject:",
    "☁️": ":cloud:",
}


def on_page_markdown(
    markdown: str,
    page: Page,
    config: MkDocsConfig,
    files: Files,
) -> str:
    """Replace emojis."""
    if page.file.name == "index" and page.title == "Home":
        for rd, md in _EMOJI_MAP.items():
            markdown = markdown.replace(rd, md)
    return markdown
