# brain-state-similarity
Hidden Markov Model-based event segmentation and event pattern similarity/recurrence

## Order of operations
### 1. Run 'HMM_event_seg.py'
Use 'run_event_seg.job' to stage on HPC.

### 2. Preprocessing
Preprocessing steps are carried out in sequence by the following scripts:
- wrangle_events.R (builds and saves intermediate file)
- evPat_similarity_proc_analyses.R (preprocesses and analyzes intermediate file)

### 3. Analyses
Representational similarity analyses are carried out in evPat_similarity_proc_analyses.R (branch coming soon)
