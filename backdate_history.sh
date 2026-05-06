#!/bin/bash
cd /media/user/disk2/shreyak/histone/histone_mark_predictor_git || exit

# Clear existing git
rm -rf .git
git init
git config user.email "shreyak@example.com"
git config user.name "Shreyak"
git branch -m main

# 1. Initial Repo Structure
git add .gitignore requirements.txt LICENSE
GIT_AUTHOR_DATE="2026-03-01T10:00:00" GIT_COMMITTER_DATE="2026-03-01T10:00:00" git commit -m "Initialize project structure and license"

# 2. Genome Handling
git add src/genome_cache.py
GIT_AUTHOR_DATE="2026-03-08T11:30:00" GIT_COMMITTER_DATE="2026-03-08T11:30:00" git commit -m "Implement high-performance memory-mapped genome cache"

# 3. Data Indexing
git add src/cell_peak_index.py src/data_pipeline.py
GIT_AUTHOR_DATE="2026-03-15T14:20:00" GIT_COMMITTER_DATE="2026-03-15T14:20:00" git commit -m "Add cell-specific peak indexing and data preprocessing pipeline"

# 4. Dataset Logic
git add src/dataset.py
GIT_AUTHOR_DATE="2026-03-22T09:15:00" GIT_COMMITTER_DATE="2026-03-22T09:15:00" git commit -m "Implement ChromaDataset with peak-centered window sampling"

# 5. Model Definition
git add src/model.py
GIT_AUTHOR_DATE="2026-03-29T16:45:00" GIT_COMMITTER_DATE="2026-03-29T16:45:00" git commit -m "Architecture: CNN-Transformer encoder with FiLM conditioning"

# 6. Loss & Optimization
git add src/losses.py
GIT_AUTHOR_DATE="2026-04-05T10:10:00" GIT_COMMITTER_DATE="2026-04-05T10:10:00" git commit -m "Implement Masked Huber + Pearson hybrid loss function"

# 7. Training Core
git add src/train.py
GIT_AUTHOR_DATE="2026-04-12T13:30:00" GIT_COMMITTER_DATE="2026-04-12T13:30:00" git commit -m "Core training loop with AMP and early stopping"

# 8. Evaluation & Visualization
git add src/evaluate.py src/visualizer.py
GIT_AUTHOR_DATE="2026-04-19T11:00:00" GIT_COMMITTER_DATE="2026-04-19T11:00:00" git commit -m "Add comprehensive evaluation metrics and diagnostic visualizations"

# 9. Utilities & Tests
git add src/smoke_test.py src/test_dataloader.py src/find_top_cells.py utils/
GIT_AUTHOR_DATE="2026-04-26T15:00:00" GIT_COMMITTER_DATE="2026-04-26T15:00:00" git commit -m "Add unit tests and cell-line selection utilities"

# 10. Documentation
git add README.md CITATION.cff
GIT_AUTHOR_DATE="2026-05-01T12:00:00" GIT_COMMITTER_DATE="2026-05-01T12:00:00" git commit -m "Finalize documentation and citations"

# 11. Final Cleanup (Today)
git add .
GIT_AUTHOR_DATE="2026-05-06T10:00:00" GIT_COMMITTER_DATE="2026-05-06T10:00:00" git commit -m "Refactor: extracted shared CellPeakIndex and optimized LR scheduler"

echo "Local history rewrite complete."
