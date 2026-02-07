# Interactive Task Gallery

Try each bead task interface below. Every demo is a live jsPsych experiment running in your browser. Examples use stimuli drawn from Aaron Steven White's psycholinguistics research, including MegaAcceptability, MegaVeridicality, Semantic Proto-Roles, and the Diverse Natural Language Inference Corpus.

<script>
window.addEventListener('message', e => {
  if (e.data?.type === 'bead-resize') {
    for (const f of document.querySelectorAll('iframe')) {
      if (f.contentWindow === e.source) { f.style.height = (e.data.height + 24) + 'px'; break; }
    }
  }
});
</script>

---

## Judgment Tasks

### Likert Rating Scale

Rate a sentence on a discrete scale with labeled endpoints. This example tests the verb *hope* in an NP-to-VP raising frame from [MegaAcceptability](https://megaattitude.io) (White & Rawlins, 2016). The dataset tests every English clause-embedding verb across syntactic frames using generic NPs like *someone* and *something*.

=== "Demo"

    <iframe src="../../gallery/demos/rating-likert.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item

    item = create_ordinal_scale_item(
        text="Someone hoped someone to leave.",
        prompt="How acceptable is this sentence?",
        scale_bounds=(1, 7),
        scale_labels={
            1: "Completely unacceptable",
            4: "Neutral",
            7: "Completely acceptable",
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-rating",
      "prompt": "How acceptable is this sentence?",
      "scale_min": 1,
      "scale_max": 7,
      "scale_labels": {
        "1": "Completely unacceptable",
        "4": "Neutral",
        "7": "Completely acceptable"
      },
      "metadata": {"verb": "hope", "frame": "NP_to_VP"}
    }
    ```

### Slider Rating

Continuous rating on a slider scale. This example tests the factive verb *forget* from [MegaVeridicality](https://megaattitude.io) (White & Rawlins, 2018). The task asks whether the embedded event (someone leaving) actually happened given the matrix verb.

=== "Demo"

    <iframe src="../../gallery/demos/rating-slider.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item

    item = create_ordinal_scale_item(
        text="Someone forgot that someone left.",
        prompt="Based on this sentence, did someone leave?",
        scale_bounds=(0, 100),
        scale_labels={
            0: "Certainly did not happen",
            100: "Certainly happened",
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-slider-rating",
      "prompt": "Based on this sentence, did someone leave?",
      "slider_min": 0,
      "slider_max": 100,
      "slider_start": 50,
      "labels": ["Certainly did not happen", "Certainly happened"],
      "metadata": {"verb": "forget", "frame": "that_S"}
    }
    ```

### Forced Choice

Choose between two alternatives. This example contrasts *want* (which permits NP-to-VP) against *hope* (which does not) from [MegaAcceptability](https://megaattitude.io) (White & Rawlins, 2016).

=== "Demo"

    <iframe src="../../gallery/demos/forced-choice.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.forced_choice import create_forced_choice_item

    item = create_forced_choice_item(
        alternatives=[
            "Someone wanted someone to leave.",
            "Someone hoped someone to leave.",
        ],
        prompt="Which sentence sounds more acceptable?",
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-forced-choice",
      "prompt": "Which sentence sounds more acceptable?",
      "alternatives": [
        "Someone <em>wanted</em> someone to leave.",
        "Someone <em>hoped</em> someone to leave."
      ],
      "metadata": {"verbs": ["want", "hope"], "frame": "NP_to_VP"}
    }
    ```

### Binary Judgment

Yes/No acceptability judgment. This example tests the verb *persuade* in an NP-to-VP object-control frame from [MegaAcceptability](https://megaattitude.io) (White & Rawlins, 2016).

=== "Demo"

    <iframe src="../../gallery/demos/binary-choice.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.binary import create_binary_item

    item = create_binary_item(
        text="Someone persuaded someone to leave.",
        prompt="Is this sentence acceptable?",
        options=["Acceptable", "Unacceptable"],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-binary-choice",
      "prompt": "Is this sentence acceptable?",
      "stimulus": "Someone <em>persuaded</em> someone to leave.",
      "choices": ["Acceptable", "Unacceptable"],
      "metadata": {"verb": "persuade", "frame": "NP_to_VP"}
    }
    ```

### Categorical Classification

Select one category from an unordered set. This example uses a factivity recast from the [Diverse Natural Language Inference Corpus](https://decomp.io) (White et al., 2018). DNC recasts existing annotations (FrameNet, factuality, etc.) into NLI premise-hypothesis format.

=== "Demo"

    <iframe src="../../gallery/demos/categorical.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.categorical import create_categorical_item

    item = create_categorical_item(
        text=(
            "Premise: The doctor managed to treat the patient.\n"
            "Hypothesis: The patient was treated."
        ),
        prompt="What is the relationship between the premise and hypothesis?",
        categories=["Entailment", "Neutral", "Contradiction"],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-categorical",
      "prompt": "What is the relationship between the premise and hypothesis?",
      "categories": ["Entailment", "Neutral", "Contradiction"],
      "metadata": {"recast_type": "factivity"}
    }
    ```

### Magnitude Estimation

Enter a numeric value with optional bounds and unit. This example uses magnitude estimation for acceptability, testing the verb *believe* in an NP-to-be-NP frame from [MegaAcceptability](https://megaattitude.io) (White & Rawlins, 2016).

=== "Demo"

    <iframe src="../../gallery/demos/magnitude.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.magnitude import create_magnitude_item

    item = create_magnitude_item(
        text="Someone believed someone to be a fool.",
        prompt="On a scale of 0 to 100, how acceptable is this sentence?",
        input_min=0,
        input_max=100,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-magnitude",
      "prompt": "On a scale of 0 to 100, how acceptable is this sentence?",
      "input_min": 0,
      "input_max": 100,
      "step": 1,
      "metadata": {"verb": "believe", "frame": "NP_to_be_NP"}
    }
    ```

### Free Text Response

Open-ended text response, single-line or multiline. This example elicits event structure descriptions for the verb *remember* in a to-VP frame, following the decomposition methodology of [UDS](https://decomp.io) (White et al., 2016).

=== "Demo"

    <iframe src="../../gallery/demos/free-text.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.free_text import create_free_text_item

    item = create_free_text_item(
        text="Someone remembered to leave.",
        prompt="What event, if any, does this sentence describe?",
        multiline=True,
        min_length=5,
        max_length=200,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-free-text",
      "prompt": "What event, if any, does this sentence describe?",
      "multiline": true,
      "rows": 3,
      "min_length": 5,
      "max_length": 200,
      "metadata": {"verb": "remember", "frame": "to_VP"}
    }
    ```

---

## Selection Tasks

### Cloze (Fill-in-the-Blank)

Dropdown selection for fill-in-the-blank gaps. This example tests clause-embedding verb frame selection from [MegaAcceptability](https://megaattitude.io) (White & Rawlins, 2016). The verb options include factive (*knew*), non-factive (*believed*), and implicative (*managed*) verbs.

=== "Demo"

    <iframe src="../../gallery/demos/cloze-dropdown.html" width="100%" height="150" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.cloze import create_cloze_item

    item = create_cloze_item(
        text="Someone {{verb1}} that someone left and {{verb2}} to go.",
        constraints={
            "verb1": ["knew", "believed", "forgot", "hoped", "denied", "doubted"],
            "verb2": ["wanted", "managed", "tried", "decided", "refused", "failed"],
        },
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-cloze-multi",
      "text": "Someone %% that someone left and %% to go.",
      "fields": [
        {"type": "dropdown", "options": ["knew", "believed", "forgot", "hoped", "denied", "doubted"]},
        {"type": "dropdown", "options": ["wanted", "managed", "tried", "decided", "refused", "failed"]}
      ],
      "require_all": true
    }
    ```

### Multi-Select

Select one or more options from a set using checkboxes. This example uses the nine proto-role properties from [Semantic Proto-Roles](https://decomp.io) (Reisinger et al., 2015) applied to the predicate *broke*. Annotators select which properties apply to each argument.

=== "Demo"

    <iframe src="../../gallery/demos/multi-select.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.multi_select import create_multi_select_item

    item = create_multi_select_item(
        text="Someone broke something.",
        prompt='Which properties apply to "someone" (arg0)?',
        options=[
            "instigation: caused the event",
            "volition: chose to be involved",
            "sentience: was aware of being involved",
            "change of state: changed state as a result",
            "existed before: existed before the event",
            "existed after: existed after the event",
            "change of location: changed location",
            "stationary: was stationary during the event",
            "physical contact: made physical contact",
        ],
        min_selections=1,
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-multi-select",
      "prompt": "Which properties apply to \"someone\" (arg0)?",
      "options": [
        "instigation: caused the event",
        "volition: chose to be involved",
        "sentience: was aware of being involved",
        "change of state: changed state as a result",
        "existed before: existed before the event",
        "existed after: existed after the event",
        "change of location: changed location",
        "stationary: was stationary during the event",
        "physical contact: made physical contact"
      ],
      "metadata": {"predicate": "broke", "argument": "arg0"}
    }
    ```

---

## Span Annotation

### Interactive Span Labeling (Fixed Labels)

Select token ranges and assign labels from a searchable fixed set. Type to filter labels or use keyboard shortcuts 1-9.

=== "Demo"

    <iframe src="../../gallery/demos/span-interactive.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item

    item = create_interactive_span_item(
        text="The committee unanimously approved the new budget proposal "
             "after reviewing the evidence.",
        prompt="Select and label semantic roles.",
        label_set=[
            "Agent", "Patient", "Theme", "Experiencer",
            "Instrument", "Beneficiary", "Location", "Time",
            "Manner", "Cause", "Purpose", "Source",
            "Goal", "Stimulus", "Result", "Predicate",
        ],
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["The", "committee", "unanimously", "approved", "the",
                 "new", "budget", "proposal", "after", "reviewing",
                 "the", "evidence", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "fixed",
        "labels": ["Agent", "Patient", "Theme", "Experiencer",
                   "Instrument", "Beneficiary", "Location", "Time",
                   "Manner", "Cause", "Purpose", "Source",
                   "Goal", "Stimulus", "Result", "Predicate"]
      }
    }
    ```

### Wikidata Entity Labeling

Interactive span labeling with Wikidata autocomplete search for labels. Select entities and search Wikidata to link them.

=== "Demo"

    <iframe src="../../gallery/demos/span-wikidata.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item

    item = create_interactive_span_item(
        text="Albert Einstein developed the theory of relativity "
             "at the Institute for Advanced Study in Princeton.",
        prompt="Select entities and search Wikidata to assign labels.",
        label_source="wikidata",
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["Albert", "Einstein", "developed", "the", "theory",
                 "of", "relativity", "at", "the", "Institute",
                 "for", "Advanced", "Study", "in", "Princeton", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "wikidata",
        "wikidata_language": "en"
      }
    }
    ```

---

## Composite Tasks

Span highlights work as an orthogonal overlay on any existing task type. The same item can have both span annotations and a rating scale, forced choice, or binary judgment.

### Span + Likert Rating

SPR change-of-state property rating with highlighted arguments. From [Semantic Proto-Roles](https://decomp.io) (Reisinger et al., 2015). Annotators rate individual proto-role properties on a Likert scale for each predicate-argument pair.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-rating.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_ordinal_scale_item(
        text="Someone broke something.",
        prompt='How likely is it that arg1 ("something") changed state?',
        scale_bounds=(1, 5),
        scale_labels={1: "Very unlikely", 3: "Neutral", 5: "Very likely"},
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[0])],
             label=SpanLabel(label="arg0")),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[2])],
             label=SpanLabel(label="arg1")),
    ])
    ```

### Span + Slider Rating

Veridicality inference with highlighted predicate and embedded clause. The factive verb *confirm* from [MegaVeridicality](https://megaattitude.io) (White & Rawlins, 2018) presupposes the truth of the embedded event.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-slider.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.ordinal_scale import create_ordinal_scale_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_ordinal_scale_item(
        text="Someone confirmed that someone left.",
        prompt="Based on this sentence, did someone leave?",
        scale_bounds=(0, 100),
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[1])],
             label=SpanLabel(label="predicate")),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[3, 4])],
             label=SpanLabel(label="embedded clause")),
    ])
    ```

### Span + Forced Choice

Compare the instigation property across predicates. From [Semantic Proto-Roles](https://decomp.io) (Reisinger et al., 2015): *threw* has high instigation for arg0, while *received* has low instigation.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-choice.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.forced_choice import create_forced_choice_item

    item = create_forced_choice_item(
        alternatives=[
            "Someone threw something.",
            "Someone received something.",
        ],
        prompt="In which sentence is arg0 more likely to have caused the event?",
    )
    ```

### Span + Binary Judgment

SPR volition property with highlighted arguments in a ditransitive frame. From [Semantic Proto-Roles](https://decomp.io) (Reisinger et al., 2015). The three-argument predicate *gave* lets annotators judge whether arg0 chose to be involved.

=== "Demo"

    <iframe src="../../gallery/demos/span-with-binary.html" width="100%" height="200" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.binary import create_binary_item
    from bead.items.span_labeling import add_spans_to_item
    from bead.items.spans import Span, SpanLabel, SpanSegment

    item = create_binary_item(
        text="Someone gave something to someone.",
        prompt=(
            'Does arg0 ("someone") have the property volition: '
            "did they choose to be involved in this event?"
        ),
        options=["Yes", "No"],
    )

    item = add_spans_to_item(item, spans=[
        Span(span_id="span_0",
             segments=[SpanSegment(element_name="text", indices=[0])],
             label=SpanLabel(label="arg0")),
        Span(span_id="span_1",
             segments=[SpanSegment(element_name="text", indices=[2])],
             label=SpanLabel(label="arg1")),
        Span(span_id="span_2",
             segments=[SpanSegment(element_name="text", indices=[4])],
             label=SpanLabel(label="arg2")),
    ])
    ```

---

## Relation Annotation

### Span Relations (Fixed Labels)

Interactive span and relation annotation with searchable fixed label sets. Create spans, then use "Add Relation" to draw directed relations between them. From UDS Semantic Role Labeling.

=== "Demo"

    <iframe src="../../gallery/demos/span-relations-fixed.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item
    from bead.items.spans import SpanSpec

    item = create_interactive_span_item(
        text="The scientist presented the findings to the committee "
             "at the annual conference.",
        prompt="Create spans and relations for semantic role labeling.",
        label_set=[
            "Agent", "Patient", "Theme", "Recipient",
            "Instrument", "Location", "Time", "Predicate",
            "Stimulus", "Goal",
        ],
        span_spec=SpanSpec(
            interaction_mode="interactive",
            label_source="fixed",
            enable_relations=True,
            relation_label_source="fixed",
            relation_labels=[
                "ARG0", "ARG1", "ARG2", "ARG3",
                "ARG-LOC", "ARG-TMP", "ARG-MNR",
                "ARG-PRP", "ARG-CAU",
            ],
            relation_directed=True,
        ),
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["The", "scientist", "presented", "the", "findings",
                 "to", "the", "committee", "at", "the", "annual",
                 "conference", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "fixed",
        "labels": ["Agent", "Patient", "Theme", "Recipient",
                   "Instrument", "Location", "Time", "Predicate",
                   "Stimulus", "Goal"],
        "enable_relations": true,
        "relation_label_source": "fixed",
        "relation_labels": ["ARG0", "ARG1", "ARG2", "ARG3",
                            "ARG-LOC", "ARG-TMP", "ARG-MNR",
                            "ARG-PRP", "ARG-CAU"],
        "relation_directed": true
      }
    }
    ```

### Span Relations (Wikidata)

Interactive entity linking and relation annotation with Wikidata search for both entity and relation labels. Useful for knowledge graph construction.

=== "Demo"

    <iframe src="../../gallery/demos/span-relations-wikidata.html" width="100%" height="250" style="border:1px solid #e0e0e0;border-radius:6px;transition:height 0.15s ease"></iframe>

=== "Python"

    ```python
    from bead.items.span_labeling import create_interactive_span_item
    from bead.items.spans import SpanSpec

    item = create_interactive_span_item(
        text="Marie Curie was born in Warsaw and later became "
             "a professor at the University of Paris.",
        prompt="Link entities via Wikidata and draw relations between them.",
        label_source="wikidata",
        span_spec=SpanSpec(
            interaction_mode="interactive",
            label_source="wikidata",
            enable_relations=True,
            relation_label_source="wikidata",
            relation_directed=True,
        ),
    )
    ```

=== "Trial JSON"

    ```json
    {
      "type": "bead-span-label",
      "tokens": {
        "text": ["Marie", "Curie", "was", "born", "in", "Warsaw",
                 "and", "later", "became", "a", "professor", "at",
                 "the", "University", "of", "Paris", "."]
      },
      "span_spec": {
        "interaction_mode": "interactive",
        "label_source": "wikidata",
        "enable_relations": true,
        "relation_label_source": "wikidata",
        "relation_directed": true,
        "wikidata_language": "en"
      }
    }
    ```
