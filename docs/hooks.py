import logging
from typing import Dict, Any

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page
from mkdocs.structure.files import Files

logger = logging.getLogger("mkdocs")

config_updated = False


# Ideally there would be a better hook, but it's unclear if context can
# be obtained any earlier
def on_template_context(
    context: Dict[str, Any],
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
    if page.file.name == "index" and page.title == "Home":
        for rd, md in _EMOJI_MAP.items():
            markdown = markdown.replace(rd, md)
    return markdown
