# VerdictEnv

A minimal, **runnable** [OpenEnv](https://huggingface.co/docs/trl/main/openenv)-style environment: a turn-taking “trial” in which a defense policy chooses **present evidence**, **object**, or **pass**; a toy **jury sentiment** vector updates; the **reward** is the change in a scalar “case score” built from that sentiment. This is meant as a **hackathon v0** you can grow toward TRL / Unsloth training and Hugging Face Spaces deployment.

## Install

```bash
cd verdict_env
pip install -e .
```

## Run a random baseline (no server)

```bash
verdict-infer
# or
python -m verdict_env.inference --task easy --seed 1
```

## Run the server (HTTP + OpenEnv `EnvClient` WebSocket)

```bash
verdict-server
# Defaults: http://0.0.0.0:8000
```

In another shell, talk to the server:

```bash
python -c "
import asyncio
from verdict_env.client import VerdictEnv
from verdict_env.models import VerdictAction
async def main():
    async with VerdictEnv('http://127.0.0.1:8000') as e:
        r = await e.reset(task='hard', seed=0)
        a = VerdictAction(action_type='pass')
        r = await e.step(a)
        s = await e.state()
        print('reward', r.reward, 'state', s)
asyncio.run(main())
"
```

## Docker (local / registry pull patterns)

```bash
docker build -t verdict_env .
docker run -p 7860:7860 -e PORT=7860 verdict_env
```

## OpenEnv notes

- **Models** live in `verdict_env/models.py` and subclass OpenEnv’s `Action` / `Observation` / `State`.
- The server uses `openenv.core.env_server.http_server.create_app` (same path as the Meta OpenEnv examples) so you get the standard **HTTP** and **`/ws`** client protocol.
- See `openenv.yaml` for task metadata and grading entry points; `verdict_env/tasks.py` encodes **easy / medium / hard** via **evidence count** and horizon.

## License

This scaffold is for the hackathon; add your own `LICENSE` when you publish.
