# Independent VoiceChat saved-capture source review: HOLD

19 November 2026. Reviewed source commit `e7d3a2d` without model weights, inference, or external calls. This is a review of the saved-data codec only; it does not grant live VoiceChat parity use.

## Verified

- Six no-model tests pass. `py_compile` and `git diff --check e7d3a2d^ e7d3a2d` pass.
- Nested torch/NumPy arrays, exact shapes and dtypes, bfloat16 raw 16-bit values, byte strings, tuple/list distinction, and dictionary insertion order survive a save/load roundtrip. An independent bfloat16 check included a NaN bit pattern and negative zero; raw `uint16` views matched.
- The manifest and each `.npy` have SHA-256 checks. A same-size one-byte tensor mutation fails with `Capture tensor hash differs`. Extra files, linked *children*, and files exceeding the byte cap are rejected by tests. The saved-data comparator test reloads the on-disk capture after mutating its original in-memory audio.
- The module imports no NeMo or model code and loads `.npy` with `allow_pickle=False`. The writer uses `np.save(..., allow_pickle=False)`. There is no model execution in the test.

## Blocking issue

The reader does not reject a symlink at the **capture-directory root**. `_total()` checks each child with `path.is_symlink()` but never checks `directory.is_symlink()`; `load_capture()` likewise accepts the root path. Reproduction with a valid 4 KiB capped capture:

```python
real = root / "real"
receipt = save_capture({"a": np.array([1, 2], dtype=np.int16)}, real, byte_cap=4096)
alias = root / "alias"
alias.symlink_to(real, target_is_directory=True)
load_capture(alias, byte_cap=4096,
             expected_manifest_sha256=receipt["manifest_sha256"])  # succeeds
```

This defeats the stated linked-artifact boundary for the supplied capture path. The content hashes still protect these particular bytes, but the reader can follow a caller-supplied path outside the intended capture directory. Reject a linked root before enumerating files, and add a negative test. If the contract intentionally permits root links, state that narrower boundary explicitly and bind the resolved directory path in the eventual worker receipt.

## Scope after fix

The test demonstrates that a comparator *can* consume reloaded bytes; the current synthetic supervisor does not yet use this codec for its arm captures. It remains a source-only component, with live admission, real worker binding, telemetry, and independent saved-result review pending.

Reviewed file SHA-256: codec `4faac77508576f0b9caf10a5a9da409490fbf1ca610528e0d6174cad6dedf39d`; tests `81f0070a482af1e7f5e34ae14fe77247aaf02e3ebd747c8a83fdf0147885bf3e`; handoff `66039c4ee434424fd71d64e2eceb34f99e8df075c8ad9e6562cbf05f24009710`.
