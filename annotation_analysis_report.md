# Annotation Analysis for CPG Clone Evaluation

## Dataset integrity

- Input file: `annotations_1000.json`
- Records: 1003
- Unique pair IDs: 1003
- Duplicate pair IDs: 0
- Populated `clone_type` labels: 1003
- Populated `manual_label` labels: 135
- Annotator IDs: 1

The evaluation should use `clone_type` as the complete ground-truth field. `manual_label` is only present for 135 records and appears to be a partial or earlier field.

## Ground-truth label distribution

| Label | N | Share | Mean sim | Median sim | Min sim | Max sim |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 9 | 0.9% | 0.9720 | 0.9696 | 0.9349 | 1.0000 |
| T2 | 281 | 28.0% | 0.9428 | 0.9380 | 0.9002 | 1.0000 |
| T3 | 320 | 31.9% | 0.9312 | 0.9259 | 0.9001 | 1.0000 |
| Not clones | 393 | 39.2% | 0.9144 | 0.9103 | 0.9000 | 0.9911 |

Binary clone prevalence is 610/1003 (60.8%) when T1/T2/T3 are treated as clones and `Not clones` as negatives.

## Related-function discovery

The annotations also support a broader retrieval interpretation: the method is not only finding strict clones, but also functions that are semantically or evolutionarily related. Counting `Functionally Equivalent`, `Functionally Related`, `Left function has added functionality`, and `Right function has added functionality` as useful related-function hits gives 510/1003 (50.8%) positive pairs. The added-functionality subtypes alone account for 157/1003 (15.7%) pairs: 68 where the left function extends the shared behavior and 89 where the right function extends it.

If the retrieval objective is "find either a clone or a meaningfully related function," then 707/1003 (70.5%) annotated pairs are useful hits. This combines the 610 strict T1/T2/T3 clone pairs with the functionally related non-clone pairs that are still valuable for program understanding, refactoring, or vulnerability-pattern search.

| Relationship annotation | N | Share | Interpretation |
| --- | --- | --- | --- |
| Functionally Equivalent | 203 | 20.2% | same behavior under renaming, formatting, or minor syntactic changes |
| Functionally Related | 150 | 15.0% | same domain task or data-flow role, but not a strict clone |
| Left function has added functionality | 68 | 6.8% | left side extends a shared base behavior |
| Right function has added functionality | 89 | 8.9% | right side extends a shared base behavior |
| Similar code structure but different functionalities | 326 | 32.5% | structural false-positive or weak relation |
| Missing relationship annotation | 167 | 16.7% | no relationship subtype recorded |

This broader view is important for a CPG paper because CPG matching should recover code with shared control/data-flow structure even when one implementation adds checks, fee routes, liquidity handling, wallet logic, or other contract-specific behavior. These are not failures for a related-code search task; they are evidence that the representation can surface meaningful variants.

| Clone type | N | Left added | Right added | Functionally related | Functionally equivalent | Related/functionality share |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 9 | 0 | 0 | 0 | 9 | 100.0% |
| T2 | 281 | 0 | 1 | 0 | 148 | 53.0% |
| T3 | 320 | 68 | 88 | 53 | 46 | 79.7% |
| Not clones | 393 | 0 | 0 | 97 | 0 | 24.7% |

## Similarity score as a binary clone ranker

The CPG similarity score has ROC-AUC 0.751 for separating T1/T2/T3 from `Not clones` within this high-similarity candidate sample. The positive and negative means are 0.9371 and 0.9144, respectively.

| Threshold | Predicted | Precision | Recall | F1 | False positives | False negatives |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | 1003 | 0.608 | 1.000 | 0.756 | 393 | 0 |
| 0.91 | 691 | 0.711 | 0.805 | 0.755 | 200 | 119 |
| 0.92 | 495 | 0.786 | 0.638 | 0.704 | 106 | 221 |
| 0.93 | 376 | 0.843 | 0.520 | 0.643 | 59 | 293 |
| 0.94 | 274 | 0.920 | 0.413 | 0.570 | 22 | 358 |
| 0.95 | 195 | 0.964 | 0.308 | 0.467 | 7 | 422 |
| 0.96 | 147 | 0.980 | 0.236 | 0.380 | 3 | 466 |
| 0.97 | 90 | 0.989 | 0.146 | 0.254 | 1 | 521 |
| 0.98 | 58 | 0.983 | 0.093 | 0.171 | 1 | 553 |
| 0.99 | 33 | 0.970 | 0.052 | 0.100 | 1 | 578 |

Best F1 in this sample occurs at threshold 0.9020 with precision 0.637, recall 0.954, and F1 0.764. Best balanced accuracy occurs at threshold 0.9265.

## Similarity score as a related-function ranker

Using the broader related-function target described above, the same thresholds give the following retrieval behavior.

| Threshold | Predicted | Precision | Recall | F1 | Unrelated/weak hits | Missed related |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | 1003 | 0.508 | 1.000 | 0.674 | 493 | 0 |
| 0.91 | 691 | 0.590 | 0.800 | 0.679 | 283 | 102 |
| 0.92 | 495 | 0.640 | 0.622 | 0.631 | 178 | 193 |
| 0.93 | 376 | 0.691 | 0.510 | 0.587 | 116 | 250 |
| 0.94 | 274 | 0.704 | 0.378 | 0.492 | 81 | 317 |
| 0.95 | 195 | 0.733 | 0.280 | 0.406 | 52 | 367 |
| 0.96 | 147 | 0.735 | 0.212 | 0.329 | 39 | 402 |
| 0.97 | 90 | 0.678 | 0.120 | 0.203 | 29 | 449 |
| 0.98 | 58 | 0.569 | 0.065 | 0.116 | 25 | 477 |
| 0.99 | 33 | 0.636 | 0.041 | 0.077 | 12 | 489 |

## Score bands

| Similarity band | N | T1 | T2 | T3 | Not clones | Clone rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90-0.92 | 508 | 0 | 91 | 130 | 287 | 43.5% |
| 0.92-0.94 | 221 | 2 | 58 | 77 | 84 | 62.0% |
| 0.94-0.96 | 127 | 0 | 49 | 59 | 19 | 85.0% |
| 0.96-0.98 | 89 | 4 | 34 | 49 | 2 | 97.8% |
| 0.98-1.00 | 58 | 3 | 49 | 5 | 1 | 98.3% |

The related-function interpretation makes the added-functionality signal visible. Relationship-only annotations peak in the 0.96-0.98 band; the 0.98-1.00 band is still highly clone-dense, but many of those pairs are T2 clone labels rather than explicit `Functionally Related` or added-functionality annotations.

| Similarity band | N | Equivalent | Functionally related | Left added | Right added | Related rate | Added-functionality rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90-0.92 | 508 | 53 | 83 | 25 | 32 | 38.0% | 11.2% |
| 0.92-0.94 | 221 | 46 | 42 | 16 | 20 | 56.1% | 16.3% |
| 0.94-0.96 | 127 | 35 | 14 | 14 | 22 | 66.9% | 28.3% |
| 0.96-0.98 | 89 | 38 | 10 | 12 | 15 | 84.3% | 30.3% |
| 0.98-1.00 | 58 | 31 | 1 | 1 | 0 | 56.9% | 1.7% |

## Function-name effect

| Group | N | T1 | T2 | T3 | Not clones | Clone rate |
| --- | --- | --- | --- | --- | --- | --- |
| same function name | 200 | 7 | 70 | 115 | 8 | 96.0% |
| different function name | 803 | 2 | 211 | 205 | 385 | 52.1% |

Same-name pairs are much cleaner candidates: 8/200 are non-clones, compared with 385/803 among different-name pairs.

## Relationship labels by clone type

| Clone type | Relationship distribution |
| --- | --- |
| T1 | Functionally Equivalent: 9 |
| T2 | Functionally Equivalent: 148; Similar code structure but different functionalities: 132; Right function has added functionality: 1 |
| T3 | Right function has added functionality: 88; Left function has added functionality: 68; Similar code structure but different functionalities: 62; Functionally Related: 53; Functionally Equivalent: 46; missing: 3 |
| Not clones | missing: 164; Similar code structure but different functionalities: 132; Functionally Related: 97 |

## Common false-positive families

These are the most frequent function-name combinations among `Not clones`.

| Left function | Right function | N |
| --- | --- | --- |
| _transfer | swapBack | 14 |
| _transfer | _transfer | 6 |
| _burn | _transfer | 5 |
| _transfer | mint | 4 |
| _transfer | _burn | 4 |
| _tokenTransfer | _transfer | 3 |
| takeBuyFee | swapBack | 3 |
| mint | _transfer | 3 |
| swapBack | _transfer | 3 |
| _tokenTransfer | _transferFrom | 3 |
| handleTax | swapBack | 2 |
| _transfer | claimToken | 2 |

## Code-size signal

| Clone type | Median left LOC | Median right LOC | Median LOC delta |
| --- | --- | --- | --- |
| T1 | 5 | 6 | 0 |
| T2 | 5 | 5 | 0 |
| T3 | 10.0 | 10.0 | 4.0 |
| Not clones | 15 | 18 | 10 |

Non-clones have the largest median line-count mismatch, while T2 pairs are usually compact and length-balanced. This is useful as a secondary diagnostic, but not sufficient as a classifier.

## Manual-label subset check

| Manual label | clone_type | N |
| --- | --- | --- |
| T3 | T3 | 43 |
| Functionally related, but not clones | Not clones | 36 |
| Not clones in any way | Not clones | 33 |
| T2 | T2 | 21 |
| Not clones in any way | T2 | 1 |
| T1 | T1 | 1 |

After normalizing `Functionally related, but not clones` to the negative class, the partial `manual_label` subset agrees with `clone_type` on 134/135 records.

## Paper-ready takeaways

1. The candidate generator has high clone yield in a difficult near-neighbor setting: 610/1003 (60.8%) annotated high-similarity pairs are T1/T2/T3 clones.
2. The score is monotonic with annotation quality. Clone rate rises from 43.5% in the 0.90-0.92 band to 98.3% above 0.98.
3. A high threshold gives high precision but low coverage. At 0.94, precision is 0.920 but recall is 0.412; at 0.96, precision is 0.980 but recall is 0.236.
4. The method finds mostly Type-2 and Type-3 clones, which is the relevant target for CPG-based structural/semantic matching: T2=281, T3=320, T1=9.
5. The related-function view is a positive result: 510/1003 (50.8%) pairs are functionally equivalent, functionally related, or added-functionality variants. If strict clones and related non-clones are both counted as useful retrieval hits, the yield is 707/1003 (70.5%). This supports positioning the approach as related-code retrieval, not just exact clone detection.
6. Added-functionality annotations are especially relevant to Type-3 clone detection: left/right added-functionality cases total 157 pairs, with 68 left-extended and 89 right-extended examples.
7. Some remaining false positives are structurally similar Solidity idioms with different semantics, especially transfer/tax/swap/mint/burn functions. This supports describing the main limitation as semantic overgeneralization among common token-contract templates.
8. Function-name agreement is a strong quality signal but not required: same-name pairs have 96.0% clone rate, while different-name pairs still contain 418 clones.
9. For a paper evaluation, present CPG similarity as a ranking/candidate-generation score rather than as a calibrated standalone classifier unless a threshold is explicitly tuned on a held-out set.
