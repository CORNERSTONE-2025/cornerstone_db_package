# Working rules for Claude Code in this repository

- Commits use the git identity configured on the machine (the owner's). Never add
  `Co-Authored-By: Claude` or any other AI attribution to commit messages or pull requests.
- Never commit local files of the owner: `cornerstone_test.py` changes, `Cornerstone_Manuale.pdf`,
  `train_config.json`, `db/`, `datastore/`.
- Code comments and docs in English; answers to the owner in Italian.
- Small steps: one change, its tests, one commit and push per step.
- Do not publish or share anything outside the repository without asking.
- Labeling semantics (agreed, do not change without asking): the prediction label is binary
  YES/NO (flare >= S_p in [t_X, t_X + W]); NO1-NO4 exist only in the stratification class
  (NO1 never, NO2 only after, NO3 before and after, NO4 only before; threshold S_s).
- Tests run in the GUI Docker image (it has every dependency):
  `docker run --rm --entrypoint "" -v $PWD:/src -w /src -e PYTHONPATH=/src cornerstone-gui:latest python -m pytest tests -q -p no:cacheprovider`
