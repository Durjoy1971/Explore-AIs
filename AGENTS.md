# Dataset work

When auditing, generating, reviewing, or balancing datasets, read
[`data/DATASET_GUIDE.md`](data/DATASET_GUIDE.md) before starting.

Preserve original datasets and existing evaluation artifacts. Write new dataset
candidates separately and report measured validation results before calling them
balanced, unique, reviewed, or ready for training.

For dataset work, use as many collaborating agents as useful and available.
Follow the two-stage workflow in the guide: first consolidate and deduplicate
existing data and report the remaining class counts; later generate new examples
only for measured training-class deficits. Do not start generation during the
initial consolidation task.

This file provides instructions; it does not automatically launch agents or run
the dataset pipeline. The workflow can be completed sequentially by one agent.
