# persona_vectors

Extraction of contrastive left/right political persona vectors used by the
activation-steering experiments in `steering/`.

The extracted vectors are included as `shared/vectors/{mistral,llama}/{left,right}_leaning_response_avg_diff.pt`,
so steering reproduces without re-running extraction. The trait-elicitation
data we wrote is in `shared/trait_data/`.

## Third-party code not included here

The extraction scripts in `shared/` (`compute_persona_vector.py`,
`extract_persona_responses.py`, `eval_coherence.py`, `eval_logprob_margin.py`,
`gen_trait_data.py`) are built on the official Persona Vectors repository and
import from it. We do not redistribute that upstream code. To re-run extraction,
clone it alongside `shared/`:

- **Persona Vectors** (Apache-2.0)
  - Repo: https://github.com/safety-research/persona_vectors
  - Pinned commit: `b8e0f044fe2410a6fad579f38324f03f13b4e917`
