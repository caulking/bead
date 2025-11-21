# Quick Start

This guide walks through creating a complete linguistic judgment experiment using the bead CLI.

**Time to complete**: 15 minutes

## Prerequisites

Ensure bead is installed with all dependencies:

```bash
pip install bead[dev,api,training]
```

Verify installation:

```bash
bead --version
```

## Example: Argument Structure Alternations

We'll build a forced-choice experiment testing verb argument structure alternations. The workflow follows bead's 6-stage pipeline: resources, templates, items, lists, deployment, and training.

## Stage 1: Resources

Create lexicons for verbs, nouns, and other lexical items.

### Import Verbs from VerbNet

Extract verbs from the VerbNet database:

```bash
bead resources import-verbnet \
    --verb-class put-9.1 \
    --output lexicons/verbs.jsonl \
    --limit 10
```

This creates a JSONL file with 10 verbs from the VerbNet put-9.1 class.

### Generate Inflected Forms

Add morphological variants using UniMorph:

```bash
bead resources import-unimorph \
    --language eng \
    --pos VERB \
    --features "V;PST" \
    --output lexicons/verbs_past.jsonl \
    --limit 10
```

### Create Noun Lexicon

Create a lexicon from a CSV file:

```bash
# First create a CSV file with columns: lemma,pos,features
echo "lemma,pos,features" > nouns.csv
echo "cat,N,{\"animacy\":\"animate\"}" >> nouns.csv
echo "book,N,{\"animacy\":\"inanimate\"}" >> nouns.csv
echo "dog,N,{\"animacy\":\"animate\"}" >> nouns.csv

bead resources create-lexicon \
    --input nouns.csv \
    --name nouns \
    --language-code eng \
    --output lexicons/nouns.jsonl
```

## Stage 2: Templates

Generate sentence templates and fill them with lexical items.

### Generate Templates

Create templates from patterns:

```bash
bead resources generate-templates \
    --from-pattern "{det} {subj} {verb} {det} {obj}" \
    --slot subj:required \
    --slot verb:required \
    --slot obj:required \
    --slot det:optional \
    --description "Basic transitive frame" \
    --output templates/basic_transitive.jsonl
```

The command auto-detects slots from `{placeholder}` syntax and creates a template with specified requirements.

### Fill Templates

Fill templates with lexical items:

```bash
bead templates fill \
    --template templates/basic_transitive.jsonl \
    --lexicon lexicons/nouns.jsonl lexicons/verbs.jsonl \
    --strategy exhaustive \
    --output filled_templates/transitive_filled.jsonl
```

The `exhaustive` strategy generates all valid combinations respecting slot constraints.

## Stage 3: Items

Create experimental items from filled templates.

### Generate Forced-Choice Pairs

Create 2-alternative forced-choice items from filled templates:

```bash
bead items create-forced-choice-from-texts \
    --texts-file filled_templates/transitive_filled.jsonl \
    --n-alternatives 2 \
    --output items/2afc_pairs.jsonl
```

This creates all possible pairs of alternatives from the filled templates.

### Validate Items

Verify items match the task type requirements:

```bash
bead items validate-for-task-type \
    --items items/2afc_pairs.jsonl \
    --task-type forced_choice
```

Expected output: `All items valid for task type: forced_choice`

## Stage 4: Lists

Partition items into experimental lists with constraints.

### Create Constraints

Define a uniqueness constraint on verbs:

```bash
bead lists create-uniqueness \
    --property-expression "item.metadata.verb" \
    --priority 5 \
    --output constraints/list_constraints.jsonl
```

This ensures each list has unique verbs (no duplicate verbs within a list).

### Partition Items

Divide items into 10 lists:

```bash
bead lists partition \
    --items items/2afc_pairs.jsonl \
    --n-lists 10 \
    --list-constraints constraints/list_constraints.jsonl \
    --strategy balanced \
    --output lists/
```

Output shows partition statistics:

```
✓ Created 10 lists
  Items per list: min=8, max=12, mean=10.0
  All constraints satisfied
```

## Stage 5: Deployment

Generate a jsPsych 8.x experiment for JATOS deployment.

### Generate Experiment

Create experiment with balanced list distribution:

```bash
bead deployment generate \
    --lists lists/ \
    --items items/2afc_pairs.jsonl \
    --experiment-type forced_choice \
    --title "Argument Structure Acceptability" \
    --instructions "Choose the more natural sentence." \
    --distribution-strategy balanced \
    --output experiment/
```

The `balanced` strategy assigns participants to the least-used list, ensuring even distribution.

### Export to JATOS

Package experiment for JATOS upload:

```bash
bead deployment export-jatos \
    --experiment experiment/ \
    --study-title "Argument Structure Study" \
    --output argument_structure.jzip
```

### Deploy to JATOS Server

Upload directly to a JATOS server:

```bash
bead deployment upload-jatos \
    --file argument_structure.jzip \
    --server https://jatos.example.com \
    --username researcher \
    --password yourpassword
```

Participants can now access the experiment via the JATOS study link.

## Stage 6: Training

Train a model on collected responses using active learning.

### Collect Data

After participants complete the experiment, download responses from JATOS:

```bash
bead training collect-data \
    --server https://jatos.example.com \
    --study-id 123 \
    --output responses/raw_responses.jsonl
```

### Train GLMM Model

Train a Generalized Linear Mixed Model with random intercepts:

```bash
bead models train-model \
    --task-type forced_choice \
    --items items/2afc_pairs.jsonl \
    --data responses/raw_responses.jsonl \
    --model-name bert-base-uncased \
    --mixed-effects-mode random_intercepts \
    --participant-intercept \
    --item-intercept \
    --output models/argument_structure_model/
```

The model accounts for participant and item variability using random effects.

### Check Convergence

Evaluate whether model performance converges to human agreement:

```bash
bead active-learning check-convergence \
    --predictions predictions.jsonl \
    --human-labels responses/raw_responses.jsonl \
    --metric krippendorff_alpha \
    --threshold 0.80
```

Expected output:

```
Krippendorff's alpha: 0.823
✓ Convergence threshold met (0.80)
```

## Next Steps

This quickstart demonstrated the basic CLI workflow. For detailed documentation:

- [Core Concepts](user-guide/concepts.md): understand stand-off annotation and the 6-stage pipeline
- [User Guide](user-guide/resources.md): explore all CLI commands for each pipeline stage
- [API Reference](api/resources.md): view complete API documentation
- [Examples Gallery](examples/gallery.md): see additional example projects

To customize your experiment:

- Add more [distribution strategies](user-guide/deployment.md#distribution-strategies) for participant assignment
- Apply [list constraints](user-guide/lists.md#constraints) for balanced experimental design
- Use [active learning](user-guide/active-learning.md) for efficient data collection
- Configure [trial presentation](user-guide/deployment.md#trial-configuration) for timing and UI settings
