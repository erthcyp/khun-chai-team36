# AGENTS.md — apps/rsu
- Edge side. Produces the 4-D vector M* and POSTs 16 bytes to the TCC.
- `MockEngine` must keep working without torch/open_clip (it powers `make dev` and the stage demo fallback).
- `ClipEngine` follows the paper: Eq. 5 prompt averaging, Eq. 6 softmax with scale 100, CPV via `semcom.cpv.verify` (tau_s=0.60, tau_delta=0.30).
- Heavy imports (torch, open_clip, PIL) stay inside functions/classes so the light install still works.
