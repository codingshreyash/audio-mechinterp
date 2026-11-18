# Independent source review: old-FIT grad-context × leaf-flag factorial

18 November 2026. Decision: GO for grad-leaf factorial source only; live model HOLD until external freeze.

I independently reviewed the committed prospective protocol, runner, launch gate, and their CPU tests from source commit `1c48480c728e925f58d0e2383e092cda0ffe4cc6`. The V1 graph/no-grad diagnostic remains a failed, consumed attempt; this source asks a distinct four-condition factorial question. This review does not approve an auditor, a freeze, or a model run.

| Gate-bound source | SHA-256 |
| --- | --- |
| `VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_PROSPECTIVE_V1.md` | `2b79d069f4cac3fa01f5a51d533cf98dc8d609e53ee571d9be0b617036533186` |
| `voxtral_old_fit_grad_leaf_factorial_runner_v1.py` | `72fa47e6310177004b613206229c2421bde34a5fd7bb27cac369697a0af0c5ab` |
| `test_voxtral_old_fit_grad_leaf_factorial_runner_v1.py` | `453b9ec149c423ca1753461d9d301354fc77df42c16ac5f2511615b3e3b2ba88` |
| `voxtral_old_fit_grad_leaf_factorial_launch_gate_v1.py` | `1283721d389c17afd6b58be54f2feeec0d34ffc9dfd84d4d88ab246199c9538f` |
| `test_voxtral_old_fit_grad_leaf_factorial_launch_gate_v1.py` | `20f287f1200dd13d16567daaaf2a236f830de7feddf80f1b7bc4cafe5d4099d9` |

The runner fixes 16 full-model forwards and 16 direct-LM forwards across `N0`, `N1`, `G0`, and `G1`, with two repeats per condition and counterbalanced site order. It normalizes the actual old-FIT `BatchFeature` mapping, validates its exact three tensor fields, dtypes, shapes, hashes, mask and audio-placeholder count before model construction. It pins the two archived clean source tensors and captures returned source, projector, LM input, LM0 pre-block, five terminal layers and first-answer logits without a cross-forward equality assertion. The hooks record actual grad context, input-leaf flags, and downstream gradient connectivity separately. Model parameters are frozen in eval mode; no VJP, generation, or attack is in the schedule.

The launch gate binds the reviewed Stage-B FP32 and V1 diagnostic failures, including all ten V1 raw files and the consumed reservation. It requires first-add committed source and auditor reviews in separate commits, then a later committed freeze and independent freeze review. Before worker construction, it creates an exclusive reservation and UTC launch record; the worker must claim that reservation. The parent enforces a 300-second process deadline, 30 GiB MPS driver limit, 1 GiB combined output limit, 8 MiB per-stream cap and separate 120-second CPU audit cap. The frozen source and freeze are checked again after model work and audit. The remaining scientific limit is one old-FIT audio case; a numerical difference would not by itself identify an audio mechanism or security consequence.

Using the pinned `models/second_architecture/.venv`, I ran the runner suite (**6/6 pass**) and gate suite (**8/8 pass**). The read-only real `--check-freeze` fails at the missing new source review after checking earlier provenance, as intended. I did not create a freeze, load model weights, or run the model. The CPU auditor needs a separate independent source review before any external freeze.
