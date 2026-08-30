#!/usr/bin/env python3
"""Shared DeepSeek API availability helpers (E1: outage protection).

check_deepseek:  quick reachability ping (a tiny chat call).
wait_until_ready: poll until reachable or timeout.

Used by pure_rollout.py (pre-flight + abort on outage) and
opid_lessons.py (pre-flight + fail-loudly).
"""

import time


def check_deepseek(env, model=None, timeout: float = 10.0) -> bool:
    """Return True if the DeepSeek API answers a 1-token ping."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=env.get("DEEPSEEK_API_KEY", ""),
            base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
            timeout=timeout,
        )
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
