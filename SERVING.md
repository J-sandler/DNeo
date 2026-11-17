# SGLang Serving

Complete [installation](INSTALL.md) first. Set `DNEO_MODEL` to the local
export directory, or to `J-sandler/Qwen3-4B-DNeo` when you have Hub access.
Use the activated environment and runtime exports from the installation page.

## Launch

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DNEO_MODEL" \
  --speculative-num-draft-tokens 16 --speculative-use-rejection-sampling \
  --trust-remote-code --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
  --disable-radix-cache --cuda-graph-backend-prefill disabled \
  --context-length 1024 --max-running-requests 256 \
  --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.60 --port 30000
```

Wait for `DNeoWorker ready:` and the server's readiness message.
`DFLASH` selects SGLang's scheduling family; the checkpoint selects DNeo's
native worker. This does not substitute a DFlash drafter.

## Chat

From another terminal:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Compute 37 times 19."}],
    "temperature": 1, "top_p": 1, "top_k": -1, "seed": 42, "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

The release supports TP=1, target temperature 0/1, draft temperature 1,
`top_p=1`, and `top_k=-1`. Its tested total context cap is 1024.
