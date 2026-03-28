# RFN And Attribution Assessment

## RFN judgment

Current verdict:
- not viable as the main Parameter Golf strategy
- viable only as a sidecar diagnostic

Reasons:
- thesis implementation is MLP-only and tiny
- transformer-specific derivations do not exist there
- no evidence yet that RFN-based rankings beat simple heuristics for compression decisions
- challenge winners are driven by engineering stacks, not explainability pipelines

## Better framing than the original thesis implementation

Anthropic-style attribution-graph thinking is a better conceptual fit for transformers than the original RFN proof-of-concept.

But:
- the full attribution-graph pipeline is too heavy for this campaign
- it still has faithfulness and coverage limitations
- it should not become the primary competition bet

## Best campaign use

Use a cheap hybrid probe:
- nodes = existing model components such as blocks, heads, or MLP subparts
- score = local relevance or indirect logit influence on concrete prompts
- compare against = weight magnitude or activation magnitude
- test = does the ranking predict pruning or quantization sensitivity better?

Continue only if the answer is yes on a small controlled experiment.
