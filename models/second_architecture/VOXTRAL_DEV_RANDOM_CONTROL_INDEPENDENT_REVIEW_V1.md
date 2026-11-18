# Independent review of posthoc Voxtral random-control diagnostic

**Verdict: descriptive audit passes; frozen development decision stays stop.** I reviewed diagnostic commit `ae0c34a` against the saved development result (`a0838b3`), CPU audit (`4d4fc6a`), and prior independent result review (`2de9bd2`). The raw result and audit SHA-256 values match the diagnostic's pins (`1c43c22720237a703441854f4e8a7fb7f41fad62bacb1be7f72d4958452e6edd` and `e79a0ae2fb09526be4830a9be1fa975459f7b9f61099e373c6113261a5490467`). I independently rehashed all 48 checkpoint/attempt pairs and parsed all 4,800 generation rows into 96 complete 50-arm directions. I did not invoke the diagnostic producer, rerun a model, or access locked-test outputs.

The frozen eligibility rule yields **87/96** directions with a valid clean recipient and a non-recipient blank baseline. For encoder 31, the equal-norm random loss arm is classified `other` in 14/96 total directions, of which **6/96** are eligible, terminal-EOS valid loss successes. For the projector, it is `other` in 65/96 total directions, of which **56/96** are eligible, terminal-EOS valid loss successes. The other eligible directions retain the recipient: 81/96 at encoder 31 and 31/96 at projector. These counts exactly match the committed diagnostic and the frozen audit's random-control successes.

Among the 56 valid projector random losses, raw output is `None` 34 times, `Nothing` 11 times, `Listen` 6 times, and five other single outputs (`Blank`, `Ole`, `Oliv`, `Swarm`, `Wulf`). The six encoder random losses are `None` three times and `Nothing`, `Grey`, and `Wreath` once each. These are **raw decoded strings**, not an independent semantic category or a new success rule. The predominance of generic projector outputs is consistent with broad output disruption under these constructed edits; it does not identify a unique word-row mechanism.

For the valid random-loss subsets, the recomputed medians are:

| Site | Clean recipient-minus-opposite score | Random effective BF16 norm | Main effective BF16 norm | Median random/main norm ratio |
| --- | ---: | ---: | ---: | ---: |
| Encoder 31 (6) | 8.460777 | 134.796524 | 134.796745 | 0.999989967 |
| Projector (56) | 9.103065 | 61.515831 | 61.516115 | 1.000000712 |

The near-equal effective norms show that the random control is not simply larger than the main edit in these subsets. They do not prove why the projector is fragile. The prospective protocol's random-control ceiling remains **18/96** per arm/site. Projector loss at **56/96** fails it, so the two-site effect-transfer gate and `development_advance_gate` remain **false**. No locked-test inference is justified by this diagnostic. These hidden-state edits are not attacker-playable audio or a security finding.
