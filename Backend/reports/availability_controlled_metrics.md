# Availability-controlled multimodal evaluation

Unrestricted rows include cases where the modality is missing, so the availability mask itself contributes signal. Controlled rows use only the cases where the modality exists, which holds the mask constant.

Mask-only baseline for comparison: **F1 0.834**.

| model | subset | n | prevalence | accuracy | F1 | specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| url | unrestricted | 700 | 0.5729 | 1.0 | 1.0 | 1.0 | 1.0 |
| html | unrestricted | 700 | 0.5729 | 0.5729 | 0.7239 | 0.0301 | 0.8374 |
| html | controlled_html_available | 325 | 0.9231 | 0.9231 | 0.9588 | 0.36 | 0.7893 |
| vision | unrestricted | 700 | 0.5729 | 0.5729 | 0.7284 | 0.0 | 0.833 |
| vision | controlled_screenshot_available | 314 | 0.9108 | 0.9108 | 0.9533 | 0.0 | 0.902 |
| fusion | unrestricted | 700 | 0.5729 | 1.0 | 1.0 | 1.0 | 1.0 |
| fusion | controlled_both_available | 308 | 0.9188 | 1.0 | 1.0 | 1.0 | 1.0 |

## How to read this

- **The URL branch alone scores 1.000.** The fused model's perfect score is inherited from it; HTML and vision add nothing measurable on top.
- **The HTML and vision models are not usable as standalone detectors.** Specificity of 0.00-0.36 means they flag most legitimate pages as phishing. Their F1 looks healthy only because phishing is ~92% of each restricted subset.
- Their ROC-AUC (0.79 HTML, 0.90 vision) shows some ranking signal, but far less than the F1 suggests.
- A controlled F1 near 0.834 for a *content* model would mean it is reading domain liveness rather than page content.

Conclusion: on this dataset the URL string features saturate the task and the multimodal pipeline cannot be shown to help.
