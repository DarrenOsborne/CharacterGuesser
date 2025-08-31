Deploying to GitHub Pages
=========================

This app is a static site: `public/` + the TF.js model (`model/`). GitHub Pages can serve it for free from the `docs/` folder on the default branch.

Quick Steps
1) Export the correct model (letters-only) so `model/model.json` exists:
   - Windows/PowerShell (venv active):
     - `python .\train\train_letters.py --epochs 12 --batch-size 256` (or your choice)
   - If you already have a `model/` with TF.js files, you can reuse it.

2) Build `docs/` for Pages:
   - `pwsh scripts/prepare_pages.ps1`
   - This copies `public/` → `docs/` and `model/` → `docs/model/`.

3) Commit and push:
   - `git add docs`
   - `git commit -m "Deploy: build docs for GitHub Pages"`
   - `git push`

4) Enable GitHub Pages:
   - Repo Settings → Pages → Source: `Deploy from a branch`
   - Branch: `main` (or your default), Folder: `/docs`
   - Save
   - Site URL: `https://<your-username>.github.io/<repo>/`

Notes
- The app loads the model via a relative path: `model/model.json` (works under the repo subpath).
- `.gitignore` ignores the local root `model/`, but explicitly allows `docs/**` and `public/model/` so you can commit deployed artifacts.
- If you see "Model not found" in the browser, ensure `docs/model/model.json` is committed and the Pages URL matches your repo path.

Switching Models
- Letters-only (recommended): run `train/train_letters.py` (exports to `model/`).
- If you experiment with other models, re-run `scripts/prepare_pages.ps1` and commit the updated `docs/model/`.

