# RSU baseline import

The K=30 prompts and offline selector come from [adaptive-clip-traffic-inference at df6ea6a](https://github.com/fongctw/adaptive-clip-traffic-inference/tree/df6ea6a73b76e4de26cc5adc93989fbe59badd0c). The inference wrapper retains normalized CLIP class prototypes and the decision code adapts the baseline's scaled softmax and confidence gates. Per-camera temporal checks and the HTTP senders implement the RoadSense contract.

## Use on Mac 1

```bash
python -m pip install -r requirements-rsu.txt
python -m unittest discover -s tests
```

Create one `rsu.vlm.ClipPredictor` to load OpenAI ViT-L/14 and cache the four class prototypes. Call `predict(frame)` with a PIL image or an OpenCV BGR frame. It returns scores in `normal_flow, congestion, accident, invalid_scene` order plus the model score, predicted class, and inference time. The first model load requires access to the model checkpoint.

Feed each prediction's `scores` to `DecisionEngine.update(camera_id, scores)` in playback order. Use the returned interval for that camera's next sample. Defaults require two consecutive reliable incident predictions for confirmation and three reliable normal predictions to exit verification. A high-scoring accident can alert on the first frame.

Start the `SemanticSender.run()` and `FrameSender.run()` workers on the asyncio event loop with an HTTP client using the TCC base URL and a finite timeout. Call `submit()` on that event loop. Run synchronous model inference in a worker thread so the HTTP workers can keep running. Count a semantic submission that returns `False` as queue overflow. Retried semantic updates preserve their frame and event IDs for receiver deduplication. Drain both queues before cancelling the workers on shutdown.

## Reproduce the prompt selection

RoadSense reads the saved `assets/prompts/ape_k30.md` ensemble. To reproduce the offline selection, install `sentence-transformers` and `scikit-learn` and run from the repo root.

```bash
python tools/prompt_selection/sbert_select_cluster_equals_k.py \
  --input tools/prompt_selection/2000prompts_gpt_5_5.md \
  --lexicon tools/prompt_selection/lexicon.md \
  --run-name gpt_5_5_cluster_eq_k30 \
  --target-counts 30 \
  --diversity-clusters 30 \
  --output-dir /tmp/roadsense-ape-k30
```

S-BERT scores each prompt against the target class's lexicon prototype and its strongest rival. K-means groups prompts within each class and the selector takes high-ranked, non-duplicate candidates across the clusters. The saved ensemble has 30 prompts per class.

## Review close scores with OpenAI

Set `OPENAI_API_KEY` on Mac 1 for this optional reviewer. Use `--openai-review`
to send JPEG evidence to OpenAI. Set `--review-roi X,Y,W,H` to crop each
evidence image to the same region, using fractions measured from the top-left
corner. Omit the crop to use the whole frame.

```bash
python -m rsu.main --video assets/demo_clips/cam01.mp4 --camera-id CAM_01 \
  --openai-review --review-roi 0.10,0.25,0.80,0.70
```

The default top-two gap is 0.08 (8 points on the 0–100 score scale), so
normal flow at 0.49 and accident at 0.48 trigger review. Tune it with
`--review-margin`. The camera samples normally at 2 seconds and switches to
1 second for verification. The worker classifies the target frame using the
previous and following samples from the same camera if they are within 3
video seconds. It uses the available evidence at the beginning or end of
the clip. There is one pending review per camera, at most two overall, and a
10-second per-camera cooldown after submitting to the API.

Match the JSON Lines `verification` record to the original prediction by
`camera_id` and `frame_id`. Use its `reviewed_class` for that frame when the
status is `resolved`. `needs_human_review` means the model abstained or the
request failed. A prediction with `review_status: "not_queued"` is still
ambiguous and has no remote verdict. Review results can arrive after later
predictions. The local temporal state and alert confirmation continue in
playback order. The RSU runner emits these records locally.
