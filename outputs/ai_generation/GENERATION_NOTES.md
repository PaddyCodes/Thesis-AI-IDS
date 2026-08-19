# AI Generation Notes

## Purpose

AI was used only to generate a bounded timing-modification plan for the
held-out malicious test records.

The AI-generated plan was not used for model training, validation,
hyperparameter selection, traditional IDS rule development, or any other
artefact-development activity.

Both intrusion detection approaches had already been trained, validated,
frozen, and evaluated on the original held-out test condition before the
AI-assisted modification plan was generated.

The AI generator was not provided with detector rules, Random Forest feature
importance, detector predictions, detector thresholds, validation results,
held-out test results, or information describing which attack families either
detector detected successfully.

---

## Input provenance

### AI modification request

SHA-256:

A5F62722AD610F173429A262CF91450062AD88B586E2EFCA5FE0E9508C8BC4AF

The request contained:

- the original held-out test dataset hash;
- allowed timing-scale bounds;
- immutable feature definitions;
- timing features eligible for modification;
- inverse-rate features;
- benign timing summary statistics;
- timing summary statistics for each attack family;
- required JSON output structure;
- restrictions preventing use of detector-specific information.

### AI generation prompt

SHA-256:

5CEB5D18B8A96539461E2100C9C756F31A3F32B063BA7102A9187DDD69B525FF

The generation prompt instructed the AI to:

- use only the supplied AI modification request;
- make no assumptions about either intrusion detector;
- avoid using IDS rules, thresholds, feature importance, model predictions,
  or detector performance;
- preserve attack labels and other immutable characteristics;
- select bounded timing-scale intervals using only the supplied timing
  statistics;
- return the result using the predefined JSON structure.

### Original held-out test dataset

SHA-256:

EB9118DA9D87345B0F13DA32B0581BE303463B99A7B6D7350C957623B612AD98

The source held-out test set contained:

- 378,120 total records;
- 314,259 benign records;
- 63,861 malicious records.

The original held-out test condition had already been evaluated before the
AI-generation stage began.

---

## Generator attempt 1

Provider: Anthropic

Model: Claude Sonnet 5

Date: 19 August 2026

Result:

The request was blocked by Anthropic's automated safeguards before a
modification plan was generated.

The interface reported that the request had been flagged by safeguards and
offered continuation using Claude Sonnet 4.6.

No timing-scale values or modification parameters were produced.

Therefore, Claude Sonnet 5 contributed no values to the final experimental
condition.

---

## Generator attempt 2

Provider: Anthropic

Model: Claude Sonnet 4.6

Date: 19 August 2026

Result:

Claude Sonnet 4.6 declined to generate the requested timing-scale intervals.

The model interpreted the requested transformation as IDS evasion because the
output would specify timing modifications for malicious network-flow records.

No timing-scale values or modification parameters were produced.

Therefore, Claude Sonnet 4.6 contributed no values to the final experimental
condition.

---

## Generator attempt 3

Provider: OpenAI

Model: OpenAI ChatGPT, GPT-5.6 Sol reasoning model

Date: 19 August 2026

Result:

The OpenAI model generated a complete detector-independent timing-morphing
plan covering all 14 attack families.

The model was provided only with the predefined AI modification request and
generation instructions.

It was not provided with:

- Random Forest feature importance;
- Random Forest predictions;
- Random Forest validation or held-out test results;
- traditional IDS rules;
- traditional IDS thresholds;
- traditional IDS validation or held-out test results;
- information identifying attack families that either detector found easy or
  difficult to detect.

The generated response followed the required JSON schema.

All generated scale intervals remained within the predefined range of
0.5 to 2.0.

No attack family was assigned a 1.0 to 1.0 interval.

The generated values were accepted without subsequent detector-informed
optimisation or manual adjustment.

---

## Generated output provenance

The raw OpenAI response was preserved as:

outputs/ai_generation/openai_raw_response.json

The operative AI modification plan was preserved as:

configs/ai_modification_plan.json

Both files have the same SHA-256:

A7A06D886B1D80F44DC18EAD8BF29E48B12AAAAD8D5442DFF64F9BC9D4D3FC20

The raw response and operative plan were programmatically checked and found
to be semantically identical.

Validation confirmed:

- plan version: 1;
- strategy: detector_independent_timing_morphing;
- attack-family coverage: 14 of 14;
- duplicate attack labels: none;
- all scale intervals within the permitted 0.5 to 2.0 range;
- min_scale less than or equal to max_scale for every family;
- no 1.0 to 1.0 no-op intervals;
- a non-empty timing-based rationale for every family.

---

## Generated timing plan

The final AI-generated timing intervals were:

| Attack family | Minimum scale | Maximum scale |
|---|---:|---:|
| Bot | 0.70 | 0.95 |
| DDoS | 0.50 | 0.70 |
| DoS GoldenEye | 0.50 | 0.60 |
| DoS Hulk | 0.50 | 0.55 |
| DoS Slowhttptest | 0.50 | 0.60 |
| DoS slowloris | 0.50 | 0.70 |
| FTP-Patator | 0.50 | 0.80 |
| Heartbleed | 0.50 | 0.70 |
| Infiltration | 0.50 | 0.60 |
| PortScan | 1.80 | 2.00 |
| SSH-Patator | 0.50 | 0.60 |
| Web Attack - Brute Force | 0.50 | 0.70 |
| Web Attack - SQL Injection | 0.50 | 0.70 |
| Web Attack - XSS | 0.50 | 0.70 |

The plan was not manually adjusted after generation.

---

## Interpretation of the AI-assisted condition

The generated condition is intended to be interpreted as an AI-assisted,
feature-space timing-morphing robustness experiment.

The AI selected bounded timing-scale intervals by comparing supplied attack
timing distributions with a benign timing reference.

Consequently, the generated plan changes timing characteristics in directions
that may make existing malicious feature vectors more similar to benign
timing behaviour.

This does not represent reconstruction of executable attacks or packet-level
network traffic.

The experiment operates only on previously labelled CIC-IDS2017 flow feature
vectors and is intended to evaluate detector robustness to AI-assisted
modification of observable timing characteristics.

Attack-family identity and ground-truth labels will remain unchanged.

---

## Planned transformation controls

At the point this generation record was created, the AI-generated plan had
not yet been applied to the held-out dataset.

The actual modification of records will be performed deterministically in
Python only after the generated plan has been validated, hashed, and committed.

The planned implementation will use:

Random seed: 42

The AI model selected only the family-specific timing-scale intervals.

The Python transformation, rather than the AI model, will be responsible for:

- selecting the individual timing scale for each malicious record;
- applying the scale consistently to eligible timing features;
- inversely adjusting dependent rate features;
- enforcing predefined transformation bounds;
- preserving immutable features;
- preserving attack-family labels;
- preserving the binary attack target;
- validating the resulting modified dataset.

Benign records will not be modified.

Neither IDS will be retrained or altered as a consequence of the AI-generated
condition.

---

## Experimental separation

The experimental sequence completed at the time this record was frozen was:

1. Dataset preprocessing was completed.
2. Training, validation, and held-out test partitions were created.
3. The Random Forest IDS was trained, tuned, validated, and frozen.
4. The traditional rule-based IDS was developed, validated, and frozen.
5. Both detector configurations were committed before final held-out testing.
6. The final held-out evaluation procedure was frozen.
7. Both frozen detectors were evaluated on the original held-out test set.
8. The original held-out results were preserved.
9. The AI modification request and generation protocol were created.
10. The AI modification request and generation protocol were frozen before
    successful AI generation.
11. Claude Sonnet 5 declined before generating any modification parameters.
12. Claude Sonnet 4.6 declined before generating any modification parameters.
13. OpenAI ChatGPT GPT-5.6 Sol generated the bounded timing-morphing plan.
14. The raw OpenAI response was preserved without manual modification.
15. The operative plan was validated against the predefined schema and
    transformation bounds.
16. The raw response and operative plan were confirmed to be identical.

At the time this generation record was frozen:

- the AI-generated plan had not yet been applied to the held-out dataset;
- no AI-modified dataset had yet been produced;
- neither IDS had been evaluated on an AI-assisted condition;
- no AI-generated or AI-modified data had been introduced into the training
  or validation partitions.

Construction and evaluation of the AI-assisted condition will occur only
after the generated plan and its provenance records have been frozen in
version control.