"""
This file contains hooks to simplify changelog generation.
"""

def define_env(env):
  'Hook function'

  @env.macro
  def gh(pr_id):
      url = f'https://github.com/mne-tools/mne-bids-pipeline/pull/{pr_id}'
      markdown = f'[#{pr_id}]({url})'
      return markdown
