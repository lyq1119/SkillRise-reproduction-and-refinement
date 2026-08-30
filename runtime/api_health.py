#!/usr/bin/env python3
"""Shared DeepSeek API availability helpers (E1: outage protection).

check_deepseek:  quick reachability ping (a tiny chat call).
wait_until_ready: poll until reachable or timeout.

Used by pure_rollout.py (pre-flight + abort on outage) and
opid_lessons.py (pre-flight + fail-loudly).
"""

import os
import time

# DeepSeek goes DIRECT, never through the (sometimes dead) local proxy.
# ALL_PROXY points at 127.0.0.1:10808 which is intermittently down; the API
# itself is reachable directly. Append to any existing NO_PROXY.
for _k in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_k, "")
    if "api.deepseek.com" not in _cur:
        os.environ[_k] = (_cur + "," if _cur else "") + "api.deepseek.com"


def make_deepseek_client(env, timeout: float = 600.0):
    """OpenAI client that ALWAYS goes DIRECT to DeepSeek.

    The shell env exports ALL_PROXY=socks5://127.0.0.1:10808 (a local SOCKS5
    proxy that is intermittently dead); httpx honors it by default, so requests
    fail in a way that looks like "DeepSeek API down". trust_env=False makes
    this client ignore every proxy env var regardless of NO_PROXY matching.
    """
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=env.get("DEEPSEEK_API_KEY", ""),
        base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        http_client=httpx.Client(trust_env=False, timeout=timeout),
    )


def check_deepseek(env, model=None, timeout: float = 10.0) -> bool:
    """Return True if the DeepSeek API answers a 1-token ping."""
    try:
        client = make_deepseek_client(env, timeout=timeout)
        client.chat.completions.create(
            model=model or env.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False


def wait_until_ready(env, model=None, timeout_seconds: float = 7200,
                     interval: float = 20.0, log=print) -> bool:
    """Poll check_deepseek until reachable. Returns True if ready,
    False if timeout_seconds elapsed first."""
    deadline = time.time() + timeout_seconds
    waited = 0.0
    while time.time() < deadline:
        if check_deepseek(env, model=model):
            if waited:
                log(f"[api_health] DeepSeek API reachable after {waited:.0f}s wait")
            return True
        time.sleep(interval)
        waited += interval
    log(f"[api_health] DeepSeek API still unreachable after {timeout_seconds:.0f}s")
    return False
