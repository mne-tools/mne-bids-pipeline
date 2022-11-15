from pathlib import Path
from typing import List

from ._logging import logger, gen_log_kwargs


CONFIG_SOURCE_PATH = Path(__file__).parent / '_config.py'


def create_template_config(
    target_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Create a template configuration file.
    """
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f'The specified path already exists: {target_path}'
        )

    # Create a template by commenting out most of the lines in _config.py
    config: List[str] = []
    with open(CONFIG_SOURCE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = (
                line if line.startswith(('#', '\n', 'import', 'from'))
                else f'# {line}'
            )
            config.append(line)

    target_path.write_text(''.join(config), encoding='utf-8')
    message = (
        f'Successfully created template configuration file at: '
        f'{target_path}'
    )
    logger.info(**gen_log_kwargs(message=message, emoji='✅', step=''))

    message = 'Please edit the file before running the pipeline.'
    logger.info(**gen_log_kwargs(message=message, emoji='💡', step=''))
