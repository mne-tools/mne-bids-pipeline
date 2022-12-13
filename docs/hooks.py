import logging
from typing import Dict, Any

from mkdocs.config.defaults import MkDocsConfig

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
