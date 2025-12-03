# Resources Module

The resources module handles Stage 1 of the pipeline: creating lexical items, templates, and constraints.

## Creating Lexicons

Lexicons store collections of lexical items (words or phrases) with linguistic features.

### From CSV

Convert CSV files to JSONL format:

```bash
bead resources create-lexicon lexicons/test_nouns.jsonl \
    --name test_nouns \
    --from-csv resources/bleached_nouns.csv \
    --language-code eng
```

The CSV file must include a `lemma` column. Optional columns: `pos`, `form`, and any features.

### Validating Existing Lexicons

Verify lexicon format:

```bash
bead resources validate-lexicon lexicons/bleached_nouns.jsonl
```

### Validating Different Lexicons

```bash
bead resources validate-lexicon lexicons/verbnet_verbs.jsonl
```

### Listing Available Lexicons

View all lexicons in a directory:

```bash
bead resources list-lexicons --directory lexicons/
```

## Working with External Resources

The CLI supports importing from VerbNet, UniMorph, PropBank, and FrameNet. These commands require network access.

### VerbNet Import

```bash
bead resources import-verbnet \
    --verb-class put-9.1 \
    --limit 5 \
    --output lexicons/verbs.jsonl
```

### UniMorph Import

```bash
bead resources import-unimorph \
    --language-code eng \
    --pos VERB \
    --features "V;PST" \
    --limit 10 \
    --output lexicons/past_verbs.jsonl
```

### PropBank Import

```bash
bead resources import-propbank \
    --frameset eat.01 \
    --output lexicons/eat.jsonl
```

### FrameNet Import

```bash
bead resources import-framenet \
    --frame Ingestion \
    --output lexicons/ingestion.jsonl
```

## Creating Templates

Templates define sentence patterns with slots for lexical items.

### From Pattern

Generate a template from a pattern string:

```bash
bead resources generate-templates templates/transitive.jsonl \
    --pattern "{subj} {verb} {obj}" \
    --name transitive \
    --language-code eng
```

### With Slot Specifications

```bash
bead resources generate-templates templates/detailed_transitive.jsonl \
    --pattern "{det} {subj} {verb} {obj}" \
    --name detailed_transitive \
    --slot subj:true \
    --slot verb:true \
    --slot obj:true \
    --slot det:false
```

### Template Variants

Generate variations from existing templates:

```bash
bead resources generate-template-variants templates/generic_frames.jsonl templates/variants.jsonl \
    --name-pattern "{base_name}_v{index}" \
    --max-variants 3
```

## Creating Constraints

Constraints restrict which lexical items can fill template slots.

### Extensional Constraints

Whitelist specific values:

```bash
bead resources create-constraint constraints/motion.jsonl \
    --type extensional \
    --slot verb \
    --values "run,walk,jump"
```

### Intensional Constraints

Use DSL expressions for feature-based filtering:

```bash
bead resources create-constraint constraints/verbs.jsonl \
    --type intensional \
    --slot verb \
    --expression "self.pos == 'VERB'"
```

### Complex Conditions

```bash
bead resources create-constraint constraints/animate.jsonl \
    --type intensional \
    --slot noun \
    --expression "self.pos == 'NOUN' and self.features.animacy == 'animate'"
```

### Relational Constraints

Define relationships across slots:

```bash
bead resources create-constraint constraints/different.jsonl \
    --type relational \
    --relation "subject.lemma != object.lemma" \
    --description "Different arguments"
```

### Agreement Constraints

```bash
bead resources create-constraint constraints/agreement.jsonl \
    --type relational \
    --relation "subject.features.number == verb.features.number" \
    --description "Subject-verb agreement"
```

## Validation

Verify resources before using them:

```bash
bead resources validate-lexicon lexicons/bleached_nouns.jsonl
```

```bash
bead resources validate-template templates/generic_frames.jsonl
```

## Listing Resources

View available resources:

```bash
bead resources list-lexicons --directory lexicons/
```

```bash
bead resources list-templates --directory templates/
```

## Cache Management

External resource imports cache results. To force refresh:

```bash
rm -rf .cache/bead/adapters/
```

## Next Steps

Once you have lexicons and templates:

1. [Fill templates](templates.md) with lexical items
2. [Create experimental items](items.md) from filled templates
3. [Apply constraints](templates.md#constraint-satisfaction) during filling

For complete API documentation, see [bead.resources API reference](../api/resources.md).
