# Testing and done criteria
- Order of evidence: analytic test, then float64 gradcheck, then oracle test.
- Write tests before implementation. Fixed seeds.
- If a test fails: never loosen the tolerance, skip, or xfail. Explain why it fails.
- Derive tolerances (fp64 gradcheck ~1e-10, forward comparisons ~1e-5) unless the blueprint says otherwise.
- "Looks right" is not a sign-off. Numeric tests only; plots and videos are supplementary.
- Heuristic PSNR targets are guides. Report actual numbers honestly.
