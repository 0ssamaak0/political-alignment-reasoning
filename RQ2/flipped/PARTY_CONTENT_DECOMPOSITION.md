# RQ2 — Party-fixed benchmark: party vs content decomposition

Per-configuration signed bias on the party-fixed content-swap benchmark, split by
condition and decomposed into a party part and a content part. This is the table behind
the worked example in the RQ2 results (Mistral steering-right, +0.145 on the political
items moving to -0.127 on the flipped ones).

**Conditions.** Political = each party paired with its own favored policy, a believable
premise (the code calls this variant "clean"). Flipped = the same party paired with the
other side's policy, a false partisan premise. The logical form and the gold label do not
change with the swap, so a fair reasoner gives the same verdict in both.

**Signed bias** = ((R_FP - R_FN) - (L_FP - L_FN)) / N_engaged, the matched-pair metric of
the methodology, over engaged items only (those returning a clear valid or invalid
verdict). Positive = right skew, negative = left skew. Items are labeled left or right by
the party they name.

**Decomposition.** The party part is the signed bias when items are grouped by the party
they name, the skew that survives the content swap (a party-label effect). The content
part is the signed bias when items are grouped by the side of the policy content, the skew
that flips sign with the content swap (a belief-bias effect). These correspond
approximately to (political + flipped) / 2 and (political - flipped) / 2, computed directly
over all engaged items. A party-label account predicts the party part carries the effect
with no sign flip; a content account predicts the content part carries it and tracks the
induced alignment.

**Content perm p** = permutation test on the size of the content part (5000 shuffles of
the content label within cell). **Tracks lean** = the content part has the sign of the
induced alignment (left negative, right positive), shown only where the content part
exceeds 0.03 in size.

Numbers are read from `decompose_out.json` (the "full" set, all eight topics, 384
political items per configuration). Regenerate with `python3 RQ2/flipped/decompose.py`
when the local response tree is available. Values are rounded to three decimals.

## Mistral-7B-Instruct-v0.2

| Config | Induced | Political | Flipped | Pooled | Party part | Content part | Tracks lean | Content perm p | Engaged % |
|:--|:--:|--:|--:|--:|--:|--:|:--:|--:|--:|
| Base    | -  | +0.000 | +0.016 | +0.008 | +0.008 | -0.008 | -    | 0.845  | 100.0 |
| RP-L    | L  | +0.026 | +0.016 | +0.021 | +0.021 | +0.005 | ~0   | 0.933  | 100.0 |
| RP-R    | R  | +0.063 | +0.021 | +0.042 | +0.042 | +0.021 | ~0   | 0.519  | 100.0 |
| Steer-L | L  | -0.031 | -0.010 | -0.021 | -0.021 | -0.010 | ~0   | 0.789  | 100.0 |
| Steer-R | R  | +0.145 | -0.127 | +0.008 | +0.008 | +0.136 | YES  | 0.0002 |  97.7 |
| DPO-L   | L  | -0.109 | +0.060 | -0.025 | -0.025 | -0.084 | YES  | 0.010  |  95.8 |
| DPO-R   | R  | +0.047 | -0.047 | +0.000 | +0.000 | +0.047 | YES  | 0.105  | 100.0 |

## Llama-3-8B-Instruct

| Config | Induced | Political | Flipped | Pooled | Party part | Content part | Tracks lean | Content perm p | Engaged % |
|:--|:--:|--:|--:|--:|--:|--:|:--:|--:|--:|
| Base    | -  | +0.021 | +0.037 | +0.029 | +0.029 | -0.008 | -     | 0.843  | 100.0 |
| RP-L    | L  | -0.011 | +0.089 | +0.039 | +0.039 | -0.050 | YES   | 0.101  |  99.5 |
| RP-R    | R  | +0.016 | -0.021 | -0.003 | -0.003 | +0.018 | ~0    | 0.577  | 100.0 |
| Steer-L | L  | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | ~0    | 1.000  | 100.0 |
| Steer-R | R  | +0.006 | -0.025 | -0.009 | -0.009 | +0.015 | ~0    | 0.665  |  84.6 |
| DPO-L   | L  | -0.084 | +0.195 | +0.055 | +0.055 | -0.139 | YES   | 0.0002 |  99.2 |
| DPO-R   | R  | +0.126 | +0.265 | +0.190 | +0.190 | -0.056 | NO*   | 0.261  |  65.6 |

\* Llama DPO-R is a refusal artifact, not a judged double standard. Its engagement is
severely asymmetric (political 50% left vs 91% right, flipped 32% left vs 90% right), so
the pooled +0.190 is produced by one-sided refusals scored as wrong, and the content part
does not track its alignment.

## Reading notes

- **Lead exhibit.** Mistral Steer-R is the cleanest sign flip, +0.145 on political to
  -0.127 on flipped. Its content part (+0.136) dominates its party part (+0.008), and the
  content part is significant (perm p = 0.0002). This is the one worked example carried in
  the talk.
- **Content tracks alignment.** The five configurations whose content part exceeds 0.03 in
  size all agree in sign with their induced alignment: Mistral Steer-R, Mistral DPO-L,
  Mistral DPO-R, Llama RP-L, Llama DPO-L (joint sign-agreement p = 0.062).
- **Significance.** By permutation, three configurations reach p < 0.05 on the content part
  (Mistral Steer-R, Mistral DPO-L, Llama DPO-L). After correcting for testing many
  configurations at once, only Mistral Steer-R and Llama DPO-L survive.
- **Party effect is small and inconsistent.** Two configurations have a significant GLM
  party term: Mistral RP-R (+0.44, p = 0.011, toward its own side) and Llama DPO-L (+0.30,
  p = 0.011, against its own side). These are log-odds coefficients from the logistic model
  P(say valid) ~ gold + party + content, not on the signed-bias scale.
- **Llama Steer-L collapses** onto a single verdict, so its signed bias is mechanically
  zero, not evidence of even-handedness. Read it with the accuracy and engagement numbers,
  not the bias alone.
- **Base models show no content effect** (Mistral -0.008, Llama -0.008), so the benchmark
  invents no skew of its own.
