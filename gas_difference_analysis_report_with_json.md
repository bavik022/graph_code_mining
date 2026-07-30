# Gas Difference Analysis for Functionally Equivalent Pairs

## Scope

- Annotation file: `annotations_1000.json`
- Gas estimate file: `gas_estimates.json`
- Functionally equivalent pairs: 203
- Pairs matched to gas estimates on both sides: 203 (100.0%)
- Unmatched pairs: 0

Gas entries are keyed by `file.sol.Contract.function(types)` and contain integer estimates from `gas_estimates.json`. Overloaded functions were resolved by parsing the annotated source signature when needed.

## Overall Difference

| Metric | Value |
| --- | --- |
| Equal gas pairs | 102 (50.2%) |
| Median relative delta vs cheaper side | 0.0% |
| Mean relative delta vs cheaper side | 13.3% |
| 75th percentile relative delta | 14.3% |
| 90th percentile relative delta | 33.3% |
| 95th percentile relative delta | 43.5% |
| Max relative delta | 245.2% |
| Median absolute delta | 0.0 |
| Mean absolute delta | 19.2 |
| Max absolute delta | 2585 |
| Left side more expensive | 49 |
| Right side more expensive | 52 |

The typical relative gas difference is small because 102/203 (50.2%) pairs have exactly equal estimates. However, the tail is substantial: 32/203 (15.8%) pairs differ by more than 25% relative to the cheaper side.

## Relative Delta Buckets

| Relative delta vs cheaper side | Pairs | Share |
| --- | --- | --- |
| 0% | 102 | 50.2% |
| 0-5% | 16 | 7.9% |
| 5-10% | 16 | 7.9% |
| 10-25% | 37 | 18.2% |
| 25-50% | 25 | 12.3% |
| >50% | 7 | 3.4% |

## Absolute Delta Buckets

Absolute units are retained as supporting detail, but the percentage buckets above are easier to interpret across functions with very different baseline gas estimates.

| Absolute gas delta | Pairs | Share |
| --- | --- | --- |
| 0 | 102 | 50.2% |
| 1-3 | 33 | 16.3% |
| 4-10 | 35 | 17.2% |
| 11-50 | 25 | 12.3% |
| 51-100 | 5 | 2.5% |
| 101-500 | 2 | 1.0% |
| >500 | 1 | 0.5% |

## By Clone Type

| Clone type | Pairs | Median relative delta | Mean relative delta | Median abs delta | Mean abs delta |
| --- | --- | --- | --- | --- | --- |
| T2 | 148 | 0.0% | 11.0% | 0.0 | 4.9 |
| T3 | 46 | 14.0% | 23.2% | 6.0 | 68.7 |
| T1 | 9 | 0.0% | 0.8% | 0.0 | 1.0 |

## Largest Percentage Differences

| Pair ID | Clone type | Left function | Right function | Relative delta | Left gas | Right gas | Abs delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t3t4_0003216 | T2 | setMaxTxAmount | updateMaxTxnAmount | 245.2% | 145 | 42 | 103 |
| t3t4_0159422 | T3 | setMaxTxnAmount | updateMaxWalletAmount | 188.9% | 36 | 104 | 68 |
| t3t4_0188230 | T2 | setTxLimit | updateMaxBuyAmount | 174.4% | 39 | 107 | 68 |
| t3t4_0114965 | T3 | setMaxTxnAmount | setMaxTxAmount | 160.6% | 33 | 86 | 53 |
| t3t4_0294475 | T2 | updateMaxWalletAmount | updateMaxWalletAmount | 147.6% | 42 | 104 | 62 |
| t3t4_0195785 | T2 | safeTransferFrom | safeTransferFrom | 80.0% | 27 | 15 | 12 |
| t3t4_0057881 | T3 | setStructure | updateBuyFees | 73.3% | 78 | 45 | 33 |
| t3t4_0066506 | T3 | safeACRDTTransfer | withdrawACRDT | 49.2% | 7836 | 5251 | 2585 |
| t3t4_0373749 | T2 | _getValues | _getValues | 46.2% | 39 | 57 | 18 |
| t3t4_0000822 | T2 | _getTValues | _getTValues | 45.5% | 48 | 33 | 15 |
| t3t4_0008618 | T2 | updateMaxBuyAmount | setMaxWalletAmount | 43.9% | 107 | 154 | 47 |
| t3t4_0175558 | T3 | manualsend | manualsend | 40.0% | 42 | 30 | 12 |
| t3t4_0001149 | T2 | manualswap | manualswap | 40.0% | 30 | 42 | 12 |
| t3t4_0001078 | T2 | manualsend | anime1 | 40.0% | 30 | 42 | 12 |
| t3t4_0136105 | T2 | manualSend | anime1 | 40.0% | 30 | 42 | 12 |

## Match Resolution

| Resolution status | Function sides |
| --- | --- |
| unique | 384 |
| signature | 22 |

## Paper-Ready Takeaways

1. All 203 functionally equivalent annotated pairs were matched to gas estimates on both sides.
2. Gas costs are often identical for functionally equivalent pairs: 102 pairs (50.2%) have exactly equal estimates.
3. Percentage differences make the tail more visible: 32 pairs (15.8%) differ by more than 25% relative to the cheaper side.
4. The mean relative difference is 13.3%, while the 95th percentile reaches 43.5%. This shows that most equivalent pairs are close, but a small subset has meaningful implementation-cost differences.
5. This supports the evaluation claim that functionally equivalent pairs generally preserve similar gas-cost profiles, while highlighting outliers that may deserve annotation review or discussion as optimization differences.
