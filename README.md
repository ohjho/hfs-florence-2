---
title: Florence 2
emoji: 📉
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: true
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Changelog
this is clone of the original space by [gokaygokay](https://huggingface.co/spaces/gokaygokay/Florence-2),
the following changes where made:

* pinning `torch==2.40` in the requirements because of the
[`ModuleNotFoundError: No module named 'flash_attn_2_cuda'` error](https://huggingface.co/gokaygokay/Florence-2-Flux/discussions/2)
* pinning `transformers==4.45.0` in the requirements because of the
`OSError: Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.`
error, [this github issue](https://github.com/huggingface/transformers/issues/4336#issuecomment-2692630348)
recommends the pinning of transformer but
[this recommendation](https://github.com/huggingface/transformers/issues/6159#issuecomment-849844030)
seems to be more future proved.
* added `pydantic==2.10.6` to the requirements to fix the
`Error: No API Found` message in gradio based on [this suggestion](https://discuss.huggingface.co/t/error-no-api-found/146226/8)
* added the feature to output confidence score with **Object Detection** with
the function `run_example_with_score()` following the example provided from
the [official doc](https://huggingface.co/microsoft/Florence-2-large#output-confidence-score-with-object-detection)
  * the feature was added to **Open Vocabulary Detection** too but it seems like the
  `transition_beam_score` are ignored in the call to `processor.post_process_generation()`
