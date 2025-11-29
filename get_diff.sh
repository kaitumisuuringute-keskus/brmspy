git diff master...HEAD -- . \
  ':(exclude)**/*.ipynb' \
  ':(exclude)**/*.lock' \
  ':(exclude)**/package-lock.json' \
  ':(exclude)**/poetry.lock' \
  ':(exclude)**/Pipfile.lock' \
  ':(exclude)docs/**' \
  ':(exclude)notebooks/**' \
  > diff.txt