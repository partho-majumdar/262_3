# Leakage Experiment: PhiUSIIL page-derived columns vs URL string only

Generated 2026-10-01T08:46:44.074270+00:00 by `training/leakage_experiment.py`.

> This is a LEAKAGE EXPERIMENT. The engineered columns are computed by fetching the page, so they are unavailable to a system that receives only a URL string. Metrics here must not be quoted as achievable system performance; they quantify what collection-time information is worth.

## What was trained

- **Engineered-only models**: 31 columns, excluding every column derivable from the URL string itself.
- **URL-only model**: the headline `metrics_url.json` run, same split, same row ids.

### Columns deliberately excluded (URL-derived, legitimately available)

```
CharContinuationRate, DegitRatioInURL, Domain, DomainLength, HasObfuscation, IsDomainIP, IsHTTPS, LetterRatioInURL, NoOfAmpersandInURL, NoOfDegitsInURL, NoOfEqualsInURL, NoOfLettersInURL, NoOfObfuscatedChar, NoOfOtherSpecialCharsInURL, NoOfQMarkInURL, NoOfSubDomain, ObfuscationRatio, SpacialCharRatioInURL, TLD, TLDLength, URL, URLCharProb, URLLength
```

### Notable page-derived columns included in the experiment

```
URLSimilarityIndex, TLDLegitimateProb, LineOfCode, LargestLineLength, HasTitle, Title, DomainTitleMatchScore, URLTitleMatchScore, HasFavicon, Robots, IsResponsive, NoOfURLRedirect, NoOfSelfRedirect, HasDescription, NoOfPopup, NoOfiFrame, HasExternalFormSubmit, HasSocialNet, HasSubmitButton, HasHiddenFields, HasPasswordField, Bank, Pay, Crypto, HasCopyrightInfo, NoOfImage, NoOfCSS, NoOfJS, NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef
```

## Test-split results

| Model | Accuracy | Precision | Phishing recall | Specificity | F1 | ROC-AUC | PR-AUC | MCC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| engineered_mlp | 0.9995 | 0.9993 | 0.9998 | 0.9991 | 0.9995 | 1.0000 | 1.0000 | 0.9989 | 0.0004 |
| engineered_logistic_regression | 0.9991 | 0.9987 | 0.9997 | 0.9983 | 0.9992 | 1.0000 | 1.0000 | 0.9981 | 0.0008 |

| Comparison vs URL-only model | Phishing recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| URL-only (shipped) | 0.9999 | 0.9992 | 0.9995 | 0.9998 |
| Engineered MLP | 0.9998 | 0.9993 | 0.9995 | 1.0000 |
| **Delta** | **-0.0001** | | | |

## Strongest single columns (test-split AUC, |AUC - 0.5| ranked)

A single column that alone nearly separates the classes is direct evidence
of collection-time leakage.

| Column | AUC |
| --- | ---: |
| `URLSimilarityIndex` | 0.9936 |
| `NoOfCSS` | 0.9898 |
| `LineOfCode` | 0.9897 |
| `NoOfSelfRef` | 0.9829 |
| `NoOfEmptyRef` | 0.9748 |
| `NoOfJS` | 0.9722 |
| `HasSubmitButton` | 0.8969 |
| `NoOfImage` | 0.8876 |
| `LargestLineLength` | 0.8817 |
| `NoOfPopup` | 0.8574 |
| `NoOfURLRedirect` | 0.8332 |
| `HasHiddenFields` | 0.8000 |

### A caveat on the top-ranked column

`URLSimilarityIndex` is PhiUSIIL's share of *near-duplicate URLs in the corpus*.
It is built from URL strings alone, so it is not page-derived in the strict
sense, but it cannot be computed for a previously unseen URL without the
reference corpus it was counted against. It is grouped with the engineered
columns because a deployed detector cannot produce it for an arbitrary input,
and because a value of exactly 0 for a novel URL versus a populated value for
a URL already present in training is a dataset artifact rather than a property
of the URL. It is reported here as measured; it is not usable by the shipped
model. Every remaining row in the table above is a genuine page-content
measurement.

## Conclusion

The two models are within 0.0001 phishing recall of each other, so
the page-derived columns add no usable signal beyond the URL string on
this split. That is the *best* outcome for the integrity of the project:
the leak, while present in the data, does not create a fake headline gain.

The shipped URL model uses only the raw URL string plus the separately
documented handcrafted branch computed from that string alone. The engineered
columns are used **only** in this experiment.
