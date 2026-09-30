# Reuse Guide: Applying ArchGuard to a New Software Project

This guide explains how to prepare a new dataset and run the ArchGuard approach on your own software project.

## Overview

ArchGuard classifies issue reports as architectural-smell-inducing or not. To apply it to train on a new project you need:

1. **Source code** for multiple released versions
2. **Issue tracker data** (issues, merge requests, discussions) from GitLab.
3. The preprocessing pipeline to extract architectural smells and link them to issues
4. A final classification dataset for training/evaluating ML models

The full data flow:

```
Source code (per version)
  → [Fact Extraction]       → dependency RSF files
  → [ARCADE Clustering]     → component RSF files
  → [Smell Detection]       → smell JSON files
  

Issue tracker data (issues + MRs + discussions)
  → [MR Version Enrichment] → enriched issues with version labels
  → [Component Mapping]     → MR file changes → architectural components
  → [Smell-to-Issue Mapping] → links smells to issues via component overlap
  → [Classification Data Generation] → classification_data_clean.json

classification_data_clean.json
  → [Traditional ML | RoBERTa | Zero-shot LLM] → metrics
```

## Prerequisites

| Requirement                            | Used for |
|----------------------------------------|----------|
| Python 3.14.2 (conda or pip)           | All scripts |
| Java 11                                | ARCADE smell detection |
| `ARCADE_Core.jar` in project root      | Clustering + smell detection |
| SciTools Understand (license required) | Fact extraction (dependency analysis) |
| GitLab API token                       | Fetching release dates for version assignment |
| OpenAI API key (optional)              | Zero-shot LLM classification |
| GPU with CUDA (optional)               | SBERT encoding, RoBERTa fine-tuning |

Install Python dependencies:

```bash
conda env create -f src/environment.yml
conda activate di
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Step-by-Step Instructions

We use `myproject` as the placeholder name throughout. Replace it with your actual project name.

---

### Step 1: Prepare Source Code

Obtain source code checkouts for each version (release/tag) you want to analyze. You need **at least 2 versions** for smell evolution analysis.

Place them under:

```
data/subject_systems/myproject/
  myproject-1.0/    # source tree for version 1.0
  myproject-1.1/    # source tree for version 1.1
  myproject-2.0/    # source tree for version 2.0
  ...
```

The naming convention `<system>-<version>` is used throughout the pipeline to match versions across stages.

---

### Step 2: Collect Issue Tracker Data

You need three types of batch JSON files extracted from your GitLab project. Place them under `data/issues/myproject/`.

#### 2a. Issue batch files (`issues_all_issues_batch_*.json`)

For a GitHub project, an optional read-only exporter translates **explicit closing
issue links** from selected merged PRs into this same schema:

```bash
gh auth status
python -m src.integrations.github_export --repo owner/repository \
  --pull-requests 26 42 \
  --output data/issues/myproject/issues_all_issues_batch_1.json
```

It needs only Python and the authenticated `gh` CLI. It preserves merge dates,
commit author dates, renamed/deleted file paths, and GitHub provenance. It fetches
paginated commit/file responses and rejects potentially truncated lists.
Cross-repository links are reported separately because the schema uses local IIDs.
Unlinked PRs are recorded without inferring links from prose. No smell labels or
discussions are generated. Continue with release registration and classification
below; a new project still needs source versions, facts, and smell evolution.

Offline contract checks:

```bash
python -m unittest discover -s tests -p test_github_export.py -v
```

Each file contains an array of issues with their metadata and related merge requests:

```json
{
  "issues": [
    {
      "issue_data": {
        "issue_metadata": {
          "id": 12345,
          "iid": 1,
          "title": "Fix memory leak in parser",
          "description": "The XML parser leaks memory when...",
          "state": "closed",
          "created_at": "2023-01-15T10:00:00Z",
          "updated_at": "2023-02-20T14:30:00Z"
        },
        "related_merge_requests": [...]
      },
      "related_mrs": [
        {
          "mr_metadata": {
            "iid": 42,
            "source_branch": "fix-parser-leak",
            "merged_at": "2023-02-01T12:00:00Z"
          },
          "commits": [
            {
              "commit_metadata": {
                "id": "abc123def...",
                "authored_date": "2023-01-20T09:00:00Z"
              },
              "commit_diff_summary": {
                "files_summary": [
                  {"new_path": "src/parser/xml_parser.cpp"},
                  {"new_path": "src/parser/memory.h"}
                ]
              }
            }
          ]
        }
      ],
      "summary": {
        "issue_id": 1,
        "issue_title": "Fix memory leak in parser",
        "total_related_mrs": 1
      }
    }
  ],
  "metadata": { "total_issues": 100, "batch": 1 }
}
```

Key requirements:
- `iid` (project-scoped issue number) is the primary identifier
- Each issue needs `related_mrs` with commit-level file diffs (`files_summary` with `new_path`)
- Commit metadata must include `authored_date` for temporal filtering
- MR metadata must include `merged_at` for version assignment

#### 2b. Discussion batch files (`issues_all_discussions_batch_*.json`)

```json
{
  "discussions": {
    "issue_1": {
      "issue_id": 1,
      "discussion_threads": [
        {
          "text": "I think this is related to the refactoring we did...",
          "author_username": "developer1",
          "created_at": "2023-01-16T11:00:00Z"
        }
      ]
    }
  },
  "metadata": { "batch": 1 }
}
```

Discussion data is optional but enables the `combined` text source for classification (title + description + discussion).

#### 2c. Diff batch files (`issues_all_diffs_batch_*.json`)

These contain commit-level diff information used to build the dependency change checklist. The format mirrors the commit data in the issue batch files, structured for batch processing of diffs.

---

### Step 3: Register Release Dates

The pipeline assigns each MR to a software version by finding the next release after the MR merge date. You need to register your project's release dates.

Edit `src/utils/release_utils.py`:

1. Add a function that returns your project's release dates:

```python
def get_myproject_release_dates():
    """Return release dates for myproject."""
    return {
        "1.0": "2023-01-01T00:00:00Z",
        "1.1": "2023-06-15T00:00:00Z",
        "2.0": "2024-01-01T00:00:00Z",
    }
```

You can hardcode dates, or fetch them from the GitLab/GitHub API (see existing functions for `inkscape`, `stackgres`, `shepard` as examples). If using the API, place your token in `secrets/gitlab_secret.txt` or `secrets/github_secret.txt`.

2. Register it in the `system_functions` dictionary inside `get_release_dates_for_system()`:

```python
system_functions = {
    'inkscape': get_inkscape_minor_version_release_dates,
    ...
    'myproject': get_myproject_release_dates,        # add this
}
```

---

### Step 4: Run the Preprocessing Pipeline

This extracts dependencies, runs clustering, detects smells, and tracks smell evolution.

```bash
python -m src.pipeline.preprocessing_pipeline \
  --system myproject \
  --language cpp      # or java, python, etc.
```

This produces:
- `data/facts/myproject/myproject-<ver>_deps.rsf` (dependencies)
- `data/clusters/myproject/acdc/<ver>/myproject-<ver>_ACDC_clusters.rsf` (components)
- `data/smells/myproject/acdc/myproject-<ver>_smells.json` (smells per version)
- `data/smells/myproject/evolution/myproject_acdc_evolution.json` (smell evolution)

**Alternatively**, if you already have ARCAN output or another detector's results, place the smell and cluster files in the expected directory structure and skip to the evolution step.

---

### Step 5: Run the Classification Pipeline

This links smells to issues and produces the classification dataset:

```bash
python -m src.pipeline.classification_pipeline \
  --system myproject \
  --algorithm acdc \
  --issue-dir data/issues/myproject \
  --evolution-file data/smells/myproject/evolution/myproject_acdc_evolution.json \
  --output-root data-output/Output/supervised_ml_all/run/myproject_acdc
```

This runs 5 steps:
1. **Issue enrichment** -- adds `mr_version` to each MR based on release dates
2. **Dependency checklist** -- identifies commits that changed dependencies
3. **Component mapping** -- maps MR file changes to architectural components via RSF clusters
4. **Smell-to-issue mapping** -- links smell evolution events to issues through component overlap
5. **Classification data generation** -- produces `classification_data_clean.json`

The final dataset is written to:
```
data-output/Output/supervised_ml_all/run/myproject_acdc/data/classification/myproject/classification_data_clean.json
```

---

### Step 6: Update Dataset Configuration

Edit `src/classification/dataset_config.py` to register your project:

```python
PROJECT_DETECTORS = {
    "inkscape": ["acdc"],
    "shepard": ["acdc", "arcan"],
    "stackgres": ["acdc", "arcan"],
    "myproject": ["acdc"],              # add this
}

PROJECT_DIR_NAMES = {
    "inkscape": "inkscape",
    "stackgres": "ongresinc-stackgres",
    "shepard": "dlr-shepard-shepard",
    "myproject": "myproject",           # add this
}

DATASET_FOLDERS = {
    ...
    ("myproject", "acdc"): "myproject_acdc",  # add this
}
```

The directory name in `PROJECT_DIR_NAMES` must match the subdirectory under `data/classification/` created by the classification pipeline.

---

### Step 7: Run Classification Experiments

#### Traditional ML

```bash
# Run on your dataset
python -m src.classification.simple_ml \
  data-output/Output/supervised_ml_all/run/myproject_acdc/data/classification/myproject/classification_data_clean.json

# With cross-validation
python -m src.classification.simple_ml \
  data-output/Output/supervised_ml_all/run/myproject_acdc/data/classification/myproject/classification_data_clean.json \
  --use-cv --cv-folds 5 --cv-repeats 2
```

#### RoBERTa Fine-tuning

```bash
# All registered project-detector-text_source combinations
python -m src.casestudy.finetune_all

# Preview first
python -m src.casestudy.finetune_all --dry-run
```

#### Zero-Shot LLM

```bash
export OPENAI_API_KEY="your-key"
python -m src.casestudy.zero_shot_all \
  --projects myproject \
  --detectors acdc \
  --text-source description \
  --model gpt-5-mini
```

#### Cross-Project Leave-One-Out

If you have multiple projects registered:

```bash
python -m src.classification.cross_project --text-source description

# Preview
python -m src.classification.cross_project --dry-run
```

---

## Classification Data Format Reference

The final `classification_data_clean.json` is a JSON array where each element represents one issue:

```json
[
  {
    "id": 42,
    "title": "Fix memory leak in parser",
    "description": "Fix memory leak in parser\nThe XML parser leaks memory when...",
    "discussion": "[dev1]: I think this relates to...",
    "discussion_unfiltered": "[dev1]: I think this relates to...\n\n[dev2]: Fixed in v2.0",
    "label": "smell",
    "version": "1.1,2.0"
  }
]
```

| Field | Description |
|-------|-------------|
| `id` | Issue IID (project-scoped identifier) |
| `title` | Issue title |
| `description` | Title + newline + description body |
| `discussion` | Discussion threads filtered to before the first MR commit (prevents temporal leakage) |
| `discussion_unfiltered` | All discussion threads without filtering |
| `label` | `"smell"` if the issue is linked to an architectural smell in any version, `"no_smell"` otherwise |
| `version` | Comma-separated list of versions where this issue has related MRs |

**Text sources used by classifiers:**
- `description` -- uses only the `description` field
- `combined` -- concatenates `description` + `discussion`
- `combined_unfiltered` -- concatenates `description` + `discussion_unfiltered`

---

## Shortcut: Providing Classification Data Directly

If you already have labeled data (e.g., from manual annotation or another tool), you can skip the full pipeline and provide `classification_data_clean.json` directly. The minimum required format is:

```json
[
  {"description": "Issue title and description text", "label": "smell"},
  {"description": "Another issue text", "label": "no_smell"}
]
```

Place this file at:
```
data-output/Output/supervised_ml_all/run/myproject_acdc/data/classification/myproject/classification_data_clean.json
```

Then update `dataset_config.py` (Step 6) and run experiments (Step 7).

Add `discussion` and `discussion_unfiltered` fields if you want to use the `combined` text sources.

---

## Directory Structure Summary

```
project-root/
├── data/
│   ├── subject_systems/myproject/
│   │   ├── myproject-1.0/          # Source code per version
│   │   └── myproject-1.1/
│   ├── facts/myproject/
│   │   ├── myproject-1.0_deps.rsf  # Dependency facts
│   │   └── myproject-1.1_deps.rsf
│   ├── clusters/myproject/acdc/
│   │   ├── 1.0/myproject-1.0_ACDC_clusters.rsf
│   │   └── 1.1/myproject-1.1_ACDC_clusters.rsf
│   ├── smells/myproject/
│   │   ├── acdc/
│   │   │   ├── myproject-1.0_smells.json
│   │   │   └── myproject-1.1_smells.json
│   │   └── evolution/
│   │       └── myproject_acdc_evolution.json
│   └── issues/myproject/
│       ├── issues_all_issues_batch_1.json
│       ├── issues_all_discussions_batch_1.json
│       └── issues_all_diffs_batch_1.json
├── data-output/Output/supervised_ml_all/run/
│   └── myproject_acdc/
│       └── data/classification/myproject/
│           └── classification_data_clean.json
├── secrets/
│   ├── gitlab_secret.txt           # GitLab API token (optional)
│   └── github_secret.txt           # GitHub API token (optional)
└── ARCADE_Core.jar                 # ARCADE library
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No smells detected | Ensure source code has more than trivial dependencies; ACDC needs non-trivial module structure |
| Empty evolution file | You need at least 2 versions with detected smells |
| MR version is `null` | Release dates are missing or MR was merged after the last known release -- add more release dates |
| No smell-to-issue mappings | Check that RSF cluster files exist for the versions where MRs are merged; component overlap may be empty if file paths in diffs don't match RSF entries |
| Low smell count in dataset | This is expected -- architectural smells are rare. The classifiers use SMOTE/undersampling to handle class imbalance |
| `ModuleNotFoundError: No module named 'src'` | Run all commands from the project root, not from `src/` |
