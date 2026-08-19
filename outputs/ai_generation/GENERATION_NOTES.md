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

The original held-out test set contained:

- 378,120 total records;
- 314,259 benign records;
- 63,861 malicious records;
- 14 malicious attack families.

The original held-out test condition had already been evaluated using both
frozen intrusion detection approaches before the AI-generation stage began.

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
- Random Forest validation results;
- Random Forest held-out test results;
- traditional IDS rules;
- traditional IDS thresholds;
- traditional IDS validation results;
- traditional IDS held-out test results;
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

The generated condition is interpreted as an AI-assisted, feature-space
timing-morphing robustness experiment.

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

Attack-family identity and ground-truth labels remain unchanged.

---

## Transformation controls

The AI-generated plan was applied to the original held-out test partition
using the deterministic Python transformation implementation.

The transformation used:

Random seed: 42

The AI model selected only the family-specific timing-scale intervals.

For each malicious record, Python selected an individual scale from the
interval assigned to that attack family.

The same effective scale was then applied consistently to all eligible timing
features for that record.

The configured timing features were:

- Flow Duration;
- Flow IAT Mean;
- Flow IAT Std;
- Flow IAT Max;
- Flow IAT Min;
- Fwd IAT Total;
- Fwd IAT Mean;
- Fwd IAT Std;
- Fwd IAT Max;
- Fwd IAT Min;
- Bwd IAT Total;
- Bwd IAT Mean;
- Bwd IAT Std;
- Bwd IAT Max;
- Bwd IAT Min;
- Active Mean;
- Active Std;
- Active Max;
- Active Min;
- Idle Mean;
- Idle Std;
- Idle Max;
- Idle Min.

The following rate features were adjusted inversely using the same effective
per-record scale:

- Flow Bytes/s;
- Flow Packets/s;
- Fwd Packets/s;
- Bwd Packets/s.

The Python transformation was responsible for:

- selecting the individual timing scale for each malicious record;
- applying the same effective scale consistently across timing features;
- inversely adjusting dependent rate features;
- enforcing the predefined maximum Flow Duration;
- preserving all non-modifiable features;
- preserving attack-family labels;
- preserving the binary attack target;
- checking for NaN or infinite values;
- validating the resulting modified dataset.

Benign records were preserved unchanged.

Neither IDS was retrained or altered as a consequence of the AI-generated
condition.

---

## Transformation implementation notes

The transformation implementation was tested using automated unit and
regression tests before successful dataset generation.

A first local execution exposed a data-type compatibility issue because some
CIC-IDS2017 timing features were stored as integer values while multiplication
by non-integer scale factors produced floating-point values.

The transformation stopped before any output dataset was written.

The implementation was updated so that only modifiable timing and rate
features are explicitly promoted to float64 before transformation.

A regression test was added to reproduce the integer-source-column case.

A subsequent execution reached the transformation integrity checks but
identified two PortScan records whose calculated Flow Duration exceeded the
120,000,000 ceiling by approximately 1.49e-08 because of floating-point
representation.

No output dataset was written by this failed execution.

The boundary calculation was subsequently changed using `numpy.nextafter` so
that constrained scale values are moved one representable floating-point value
towards zero.

This ensures that multiplication of the effective scale by the original Flow
Duration cannot exceed the configured ceiling solely because of floating-point
rounding.

A regression test was added for this boundary condition.

Following these fixes, the complete automated test suite contained 79 tests
and all 79 passed before the successful transformation was executed.

---

## Duration constraint handling

The configured maximum Flow Duration was:

120,000,000

The majority of generated timing scales could be applied directly.

Thirteen PortScan records required their sampled scale to be reduced because
the requested expansion would otherwise have exceeded the configured maximum
Flow Duration.

For these records:

- the original AI-generated family interval was not altered;
- the originally sampled scale was retained in the transformation audit;
- only the effective per-record applied scale was reduced;
- the same effective scale was used across all timing and inverse-rate
  transformations for that record.

The transformation audit records whether each malicious record was
constrained and stores both its sampled and applied scale.

No other attack family required duration-boundary constraint handling.

---

## Generated AI-assisted dataset

The AI-assisted held-out condition was generated successfully.

Modified dataset:

data/processed/ai_modified_test.csv

SHA-256:

8BA5578F5DDCD0A11120C51C9632E21A5077452D9E3F208D77BF2876D4A268AD

The generated condition contained:

- 378,120 total records;
- 314,259 benign records;
- 63,861 malicious records;
- all 14 malicious attack families from the original held-out condition.

The generated dataset is retained locally under `data/processed/` and is not
committed to version control.

Its SHA-256 hash provides a reproducible identifier for the exact generated
experimental condition.

---

## Transformation audit

The per-record scale audit was preserved as:

outputs/ai_generation/ai_transformation_scales.csv

SHA-256:

2B9EE487BC81B1EC61CA41DFDCD40484A9531617D9C4872582A1CDF4500D7047

The audit contains one record for every malicious held-out test record and
records:

- original source row;
- attack-family label;
- sampled scale;
- applied scale;
- whether the scale was constrained.

The audit therefore contains:

63,861 records.

Thirteen records were marked as constrained.

---

## Transformation summary

A family-level transformation summary was preserved as:

outputs/ai_generation/ai_transformation_summary.csv

The successful transformation produced the following results:

| Attack family | Records | Requested interval | Mean applied scale | Constrained |
|---|---:|---:|---:|---:|
| Bot | 284 | 0.70-0.95 | 0.820688 | 0 |
| DDoS | 19,206 | 0.50-0.70 | 0.600038 | 0 |
| DoS GoldenEye | 1,463 | 0.50-0.60 | 0.551141 | 0 |
| DoS Hulk | 25,959 | 0.50-0.55 | 0.525072 | 0 |
| DoS Slowhttptest | 765 | 0.50-0.60 | 0.548590 | 0 |
| DoS slowloris | 849 | 0.50-0.70 | 0.599173 | 0 |
| FTP-Patator | 835 | 0.50-0.80 | 0.647635 | 0 |
| Heartbleed | 2 | 0.50-0.70 | 0.630364 | 0 |
| Infiltration | 2 | 0.50-0.60 | 0.538959 | 0 |
| PortScan | 13,675 | 1.80-2.00 | 1.899763 | 13 |
| SSH-Patator | 488 | 0.50-0.60 | 0.550412 | 0 |
| Web Attack - Brute Force | 232 | 0.50-0.70 | 0.591492 | 0 |
| Web Attack - SQL Injection | 4 | 0.50-0.70 | 0.602501 | 0 |
| Web Attack - XSS | 97 | 0.50-0.70 | 0.608358 | 0 |

The slight difference between the requested and mean applied PortScan scale
resulted from the 13 records affected by the maximum Flow Duration constraint.

---

## Transformation integrity

The successful transformation completed all configured integrity checks.

The transformation implementation verified that:

- the source held-out dataset hash matched the frozen source hash;
- the AI modification plan hash matched the frozen plan hash;
- the dataset row count remained unchanged;
- the benign/attack class distribution remained unchanged;
- all 14 malicious attack families remained present;
- benign records remained numerically unchanged;
- attack-family labels remained unchanged;
- binary attack targets remained unchanged;
- all non-modifiable features remained unchanged;
- timing transformations used a consistent effective scale per record;
- rate features were transformed inversely using the same scale;
- no transformed Flow Duration exceeded 120,000,000;
- no zero or negative effective timing scales were introduced;
- no NaN values were introduced;
- no infinite values were introduced.

The successful transformation completed before either intrusion detection
approach was evaluated against the AI-assisted condition.

---

## Experimental separation

The experimental sequence completed at the time this record was updated was:

1. Dataset preprocessing was completed.
2. Training, validation, and held-out test partitions were created.
3. The Random Forest IDS was trained and tuned using training data.
4. The Random Forest IDS was evaluated on validation data and frozen.
5. The traditional rule-based IDS was developed using training-derived
   characteristics.
6. The traditional IDS rules were frozen before validation exposure.
7. The traditional IDS was evaluated on validation data.
8. Both detector configurations were frozen before final held-out testing.
9. The final held-out evaluation procedure was frozen in version control.
10. Both frozen detectors were evaluated on the original held-out test set.
11. The original held-out test results were preserved.
12. The AI modification request and generation protocol were created.
13. The AI modification request and generation protocol were frozen before
    successful AI generation.
14. Claude Sonnet 5 declined before generating modification parameters.
15. Claude Sonnet 4.6 declined before generating modification parameters.
16. OpenAI ChatGPT GPT-5.6 Sol generated the bounded timing-morphing plan.
17. The raw OpenAI response was preserved without manual modification.
18. The operative AI plan was validated against the predefined schema and
    transformation bounds.
19. The raw response and operative plan were confirmed to be identical.
20. The AI-generated plan and its provenance were frozen in version control.
21. The deterministic transformation implementation was created.
22. The transformation implementation was tested before dataset generation.
23. Integer-source-column handling was corrected and covered by a regression
    test.
24. Floating-point duration-boundary handling was corrected and covered by a
    regression test.
25. All 79 automated tests passed.
26. The transformation implementation was frozen in version control before
    successful dataset generation.
27. The AI-generated plan was applied to the held-out malicious records using
    random seed 42.
28. Transformation integrity checks passed.
29. The AI-assisted held-out dataset was generated successfully.
30. The generated dataset and transformation audit were assigned reproducible
    SHA-256 hashes.
31. At the time these transformation records were frozen, neither IDS had yet
    been evaluated against the AI-assisted condition.

---

## Experimental state at this freeze point

At the time this transformation record was frozen:

- the original held-out test experiment had already been completed;
- the AI modification plan had been generated and frozen;
- the AI-assisted held-out dataset had been generated successfully;
- the AI-assisted dataset had not been used for training;
- the AI-assisted dataset had not been used for validation;
- neither IDS had been retrained;
- neither IDS had been altered;
- neither IDS had yet been evaluated against the AI-assisted condition.

The next experimental stage is evaluation of both previously frozen intrusion
detection approaches against the same AI-assisted held-out condition.

That evaluation will occur only after the transformation dataset and
associated provenance evidence have been verified and frozen.
