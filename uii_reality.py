"""
UII v16.4 — uii_reality.py
Perturbation Harnessing

Role: Reality is the authoritative, non-optimizing source of perturbations.
It executes actions and returns measured channel snapshots — nothing more.
The browser is a low-fidelity viewport into a slice of reality.

v16.4 changes:
  - measure_channels() added: pre/post channel snapshots per action
  - execute() returns (pre_channels, post_channels, context)
  - _build_channel_probes() and _CHANNEL_PROBES moved here from uii_triad
  - Channel signals flow directly to SensingOperator — no SIPA intermediary

Also contains:
  - CouplingMatrixEstimator (empirical co-movement — learns from reality)
  - BrowserRealityAdapter (Playwright-based reality interface)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Callable
import numpy as np
import copy
import time
import json
from collections import deque

import hashlib

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace,
    RealityAdapter,
    AgentHandler, AVAILABLE_AGENTS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Channel probe registry — moved from uii_triad
# ──────────────────────────────────────────────────────────────────────────────

def _build_channel_probes() -> Dict[str, Callable]:
    """
    One-time construction of the channel probe registry.
    Each entry maps a channel_id to a zero-arg callable returning
    {magnitude, rate, coverage} or None if unavailable on this substrate.

    Hardware-dependent channels (audio_in, serial_port, display, video_in) have
    no probe — they stay dark unless fed by an external adapter.
    Context-dependent channels (browser, api_llm, ssh_remote) are handled
    separately in measure_channels() since their signal derives from action
    outcomes, not system introspection.

    All probes are silent on failure — a substrate that can't supply a reading
    simply leaves that channel dark this step.
    """
    import gc as _gc
    import os as _os
    import sys as _sys

    probes: Dict[str, Callable] = {}

    # ── Always available ──────────────────────────────────────────────────────
    probes['clock']          = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['clock_rate']     = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['os_signals']     = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['entropy_source'] = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['env_vars']       = lambda: {
        'magnitude': min(len(_os.environ) / 100.0, 1.0), 'rate': 0.0, 'coverage': 1.0,
    }
    probes['gc_pressure']    = lambda: (lambda c: {
        'magnitude': min((c[0] / 700.0 + c[1] / 70.0 + c[2] / 10.0) / 3.0, 1.0),
        'rate': 1.0, 'coverage': 1.0,
    })(_gc.get_count())

    # ── stdin — non-blocking check ────────────────────────────────────────────
    def _stdin() -> Optional[Dict]:
        try:
            import select
            ready = bool(select.select([_sys.stdin], [], [], 0.0)[0])
            return {'magnitude': float(ready), 'rate': float(ready), 'coverage': 1.0}
        except Exception:
            return None
    probes['stdin'] = _stdin

    # ── psutil block — graceful no-op if not installed ────────────────────────
    try:
        import psutil as _ps
        _proc = _ps.Process()

        probes['resource_cpu']     = lambda: {'magnitude': _ps.cpu_percent() / 100.0,            'rate': 1.0, 'coverage': 1.0}
        probes['resource_memory']  = lambda: {'magnitude': _ps.virtual_memory().percent / 100.0,  'rate': 1.0, 'coverage': 1.0}
        probes['resource_swap']    = lambda: {'magnitude': _ps.swap_memory().percent / 100.0,     'rate': 1.0, 'coverage': 1.0}
        probes['resource_disk']    = lambda: {'magnitude': _ps.disk_usage('/').percent / 100.0,   'rate': 1.0, 'coverage': 1.0}
        probes['process_self']     = lambda: {'magnitude': _proc.memory_percent() / 100.0,        'rate': 1.0, 'coverage': 1.0}
        probes['process_threads']  = lambda: {'magnitude': min(_proc.num_threads() / 50.0, 1.0),  'rate': 1.0, 'coverage': 1.0}
        probes['process_children'] = lambda: {
            'magnitude': min(len(_proc.children()) / 10.0, 1.0),
            'rate':      float(len(_proc.children()) > 0),
            'coverage':  1.0,
        }

        def _fd() -> Optional[Dict]:
            try:
                return {'magnitude': min(_proc.num_fds() / 1024.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except AttributeError:
                return None
        probes['resource_fd'] = _fd

        def _thermal() -> Optional[Dict]:
            try:
                t = _ps.sensors_temperatures()
                if t:
                    vals = [x.current for readings in t.values() for x in readings]
                    return {'magnitude': min(max(vals) / 100.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['resource_thermal'] = _thermal

        def _battery() -> Optional[Dict]:
            try:
                b = _ps.sensors_battery()
                if b:
                    return {
                        'magnitude': b.percent / 100.0,
                        'rate':      float(not b.power_plugged),
                        'coverage':  1.0,
                    }
            except Exception:
                pass
            return None
        probes['resource_battery'] = _battery

        def _net_if() -> Optional[Dict]:
            try:
                stats = _ps.net_if_stats()
                up = sum(1 for s in stats.values() if s.isup)
                return {'magnitude': up / max(len(stats), 1), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                return None
        probes['network_interface'] = _net_if

        def _net_bw() -> Optional[Dict]:
            try:
                c = _ps.net_io_counters()
                if c:
                    return {'magnitude': min((c.bytes_sent + c.bytes_recv) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['network_bandwidth'] = _net_bw

        def _disk_io() -> Optional[Dict]:
            try:
                c = _ps.disk_io_counters()
                if c:
                    return {'magnitude': min((c.read_bytes + c.write_bytes) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['filesystem_io'] = _disk_io

        def _syslog() -> Optional[Dict]:
            import os as _o
            path = '/var/log/syslog'
            if _o.path.exists(path):
                return {'magnitude': min(_o.path.getsize(path) / 1e7, 1.0), 'rate': 1.0, 'coverage': 1.0}
            return None
        probes['system_logs'] = _syslog

    except ImportError:
        pass

    return probes


# Built once at import time. Probes are stateless callables, safe to share.
_CHANNEL_PROBES: Dict[str, Callable] = _build_channel_probes()


# ──────────────────────────────────────────────────────────────────────────────
# CouplingMatrixEstimator
# ──────────────────────────────────────────────────────────────────────────────

class CouplingMatrixEstimator:
    """
    Empirical S/I/P/A co-movement tracker.

    Tracks how the four substrate dimensions move together across reality interactions.
    Updates via slow exponential moving average (alpha=0.05 — long memory, resists noise).

    This is the most important Layer 2 component.
    A 4x4 matrix of empirical coupling strengths IS the basin's causal signature
    in compact heritable form.

    A bootstrapping Triad with an inherited coupling matrix starts with calibrated
    gradients — it knows how its lineage's SIPA dimensions actually couple,
    without being told explicitly.

    matrix[i][j] = how much dim_i tends to move when dim_j moves.
    Starts as identity (no assumed couplings).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.matrix = np.eye(4)  # Start as identity — no assumed couplings
        self.dims = SUBSTRATE_DIMS
        self.observation_count = 0
        self.calibration_threshold = 200  # Observations needed for full confidence
        # v16: per-action delta tracking (replaces CAM.affordance_deltas).
        # update() called from step() with executed action and SIPA before/after.
        # distill_to_ledger() reads affordance_deltas for action_substrate_map merge.
        self.affordance_deltas: Dict[str, List[Dict[str, float]]] = {}

    def update(self, action: str,
               state_before: Dict[str, float],
               state_after:  Dict[str, float]):
        """
        v16: Called from step() after every execution.
        Computes SIPA delta, updates coupling matrix, records per-action delta.

        action:       affordance name (e.g. 'navigate', 'python')
        state_before: state.as_dict() before execution
        state_after:  state.as_dict() after execution
        """
        observed_delta = {
            dim: state_after.get(dim, 0.0) - state_before.get(dim, 0.0)
            for dim in self.dims
        }
        self.observe(observed_delta)
        if action not in self.affordance_deltas:
            self.affordance_deltas[action] = []
        # Keep last 50 observations per action — bounded memory
        self.affordance_deltas[action].append(observed_delta)
        if len(self.affordance_deltas[action]) > 50:
            self.affordance_deltas[action].pop(0)

    def get_empirical_action_map(self) -> Dict[str, Dict[str, float]]:
        """
        Return mean SIPA delta per action for actions with >= 5 observations.
        Called by distill_to_ledger() for action_substrate_map merge.
        Replaces CAM.get_empirical_action_map() — no other callers.
        """
        result = {}
        for action, deltas in self.affordance_deltas.items():
            if len(deltas) < 5:
                continue
            result[action] = {
                dim: float(np.mean([d.get(dim, 0.0) for d in deltas]))
                for dim in self.dims
            }
        return result

    def observe(self, observed_delta: Dict[str, float]):
        """Update coupling matrix from observed substrate co-movement."""
        for i, dim_i in enumerate(self.dims):
            for j, dim_j in enumerate(self.dims):
                if i == j:
                    continue
                di = observed_delta.get(dim_i, 0.0)
                dj = observed_delta.get(dim_j, 0.0)
                if abs(dj) > 1e-6:
                    observed_coupling = np.clip(di / dj, -2.0, 2.0)
                    self.matrix[i][j] = (
                        (1 - self.alpha) * self.matrix[i][j] +
                        self.alpha * observed_coupling
                    )
        self.observation_count += 1

    def get_confidence(self) -> float:
        return min(1.0, self.observation_count / self.calibration_threshold)

    def to_ledger_entry(self) -> Dict:
        return {
            'matrix': self.matrix.tolist(),
            'observations': self.observation_count,
            'confidence': self.get_confidence()
        }

    @classmethod
    def from_ledger_entry(cls, entry: Dict, alpha: float = 0.05) -> 'CouplingMatrixEstimator':
        estimator = cls(alpha=alpha)
        estimator.matrix = np.array(entry['matrix'])
        estimator.observation_count = entry.get('observations', 0)
        return estimator

    @classmethod
    def merge(cls, parent_entry: Dict, session_estimator: 'CouplingMatrixEstimator') -> 'CouplingMatrixEstimator':
        """
        Merge parent coupling matrix with session observations.
        Session weighted by its own confidence — low-evidence sessions
        don't overwrite high-confidence inherited structure.
        """
        merged = cls()
        parent_matrix = np.array(parent_entry.get('matrix', np.eye(4).tolist()))
        session_conf = session_estimator.get_confidence()
        merged.matrix = (1 - session_conf) * parent_matrix + session_conf * session_estimator.matrix
        merged.observation_count = (
            parent_entry.get('observations', 0) + session_estimator.observation_count
        )
        return merged


# ============================================================
# NEW: ResidualTracker


class BrowserRealityAdapter(RealityAdapter):
    """
    Browser-based reality interface via Playwright.

    v14: response_latency_ms added to context for ResidualTracker.
    Python affordance ungated (v13.4+).
    """

    def __init__(self, base_delta: float = 0.03, headless: bool = True,
                 start_url: str = 'https://zenodo.org/records/18017374'):
        self.base_delta = base_delta
        self.headless = headless
        self.start_url = start_url

        self.initialized: bool = False
        self._ever_navigated: bool = False

        self.volatility_history: deque = deque(maxlen=10)

        from playwright.sync_api import sync_playwright
        self._init_browser()

    def _init_browser(self):
        """Initialize Playwright browser instance and navigate to start_url.

        Browser launch failure is fatal — no browser means no Reality interface.
        Navigation failure is non-fatal: the browser is functional and the CNS
        can issue navigate actions once the step loop starts. A warning is logged
        so the operator knows the starting surface is blank.
        """
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720})
        self.page = self.context.new_page()

        try:
            # v16.1: networkidle ensures JS-rendered links are present before
            # the first affordance query. domcontentloaded fires before JS
            # hydration on SPAs like Zenodo, leaving links=[] every step.
            self.page.goto(self.start_url, wait_until='networkidle', timeout=15000)
            self._ever_navigated = True
        except Exception as e:
            # networkidle timeout is non-fatal — page may still be usable.
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                self._ever_navigated = True
            except Exception:
                print(f"[REALITY] Warning: start_url navigation failed ({e}). "
                      f"Browser ready — CNS must navigate before links are available.")

        self.initialized = True

    def get_current_affordances(self) -> Dict:
        """Extract all executable actions from current DOM state.

        v16.1 changes:
        - Brief stabilization wait so JS-rendered content has time to appear.
        - Link visibility: offsetWidth > 0 || offsetHeight > 0 replaces
          offsetParent !== null. offsetParent fails for links in position:fixed,
          sticky headers, and overflow:hidden ancestors — common on modern SPAs.
          offsetWidth/Height is a direct layout measurement, correct in all cases.
        - Self-links excluded: l.url !== window.location.href.
        """
        try:
            # Brief stabilization wait — improves link detection on JS-heavy pages
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=1000)
            except Exception:
                pass

            current_url = self.page.url

            if current_url == 'about:blank' and not self._ever_navigated:
                return {
                    'links': [],
                    'buttons': [],
                    'inputs': [],
                    'readable': [],
                    'current_url': 'about:blank',
                    'page_title': '',
                    'scroll_position': 0,
                    'viewport_height': 0,
                    'total_height': 0
                }

            if current_url != 'about:blank':
                self._ever_navigated = True

            affordances = self.page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        url: a.href,
                        text: a.innerText.trim().slice(0, 100),
                        visible: (a.offsetWidth > 0 || a.offsetHeight > 0)
                    }))
                    .filter(l =>
                        l.visible &&
                        l.url.startsWith('http') &&
                        l.url !== window.location.href
                    )
                    .slice(0, 50);

                const buttons = Array.from(document.querySelectorAll(
                    'button, [role="button"], input[type="submit"], input[type="button"]'
                ))
                    .map((b, i) => ({
                        selector: b.id ? `#${b.id}` : `${b.tagName.toLowerCase()}:nth-of-type(${i+1})`,
                        text: b.innerText || b.value || '',
                        visible: b.offsetParent !== null
                    }))
                    .filter(b => b.visible)
                    .slice(0, 30);

                const inputs = Array.from(document.querySelectorAll(
                    'input:not([type="submit"]):not([type="button"]), textarea, select'
                ))
                    .map((inp, i) => ({
                        selector: inp.id ? `#${inp.id}` : inp.name ? `[name="${inp.name}"]` : `input:nth-of-type(${i+1})`,
                        type: inp.type || inp.tagName.toLowerCase(),
                        placeholder: inp.placeholder || '',
                        visible: inp.offsetParent !== null
                    }))
                    .filter(inp => inp.visible)
                    .slice(0, 20);

                const readable = Array.from(document.querySelectorAll(
                    'article, main, [role="main"], .content, #content'
                ))
                    .map((el, i) => ({
                        selector: el.id ? `#${el.id}` : el.className ? `.${el.className.split(' ')[0]}` : `article:nth-of-type(${i+1})`,
                        preview: el.innerText.slice(0, 200)
                    }))
                    .slice(0, 10);

                return {
                    links: links,
                    buttons: buttons,
                    inputs: inputs,
                    readable: readable,
                    current_url: window.location.href,
                    page_title: document.title,
                    scroll_position: window.scrollY,
                    viewport_height: window.innerHeight,
                    total_height: document.documentElement.scrollHeight
                };
            }""")

            return affordances

        except Exception as e:
            return {
                'links': [],
                'buttons': [],
                'inputs': [],
                'readable': [],
                'current_url': '',
                'page_title': '',
                'scroll_position': 0,
                'viewport_height': 0,
                'total_height': 0
            }

    def _measure_dom_state(self) -> Dict:
        """Measure current DOM state for delta calculation."""
        try:
            metrics = self.page.evaluate("""() => {
                return {
                    text_length: document.body.innerText.length,
                    link_count: document.querySelectorAll('a').length,
                    image_count: document.querySelectorAll('img').length,
                    input_count: document.querySelectorAll('input, textarea, select').length,
                    dom_depth: (function() {
                        let maxDepth = 0;
                        function getDepth(element, depth) {
                            maxDepth = Math.max(maxDepth, depth);
                            for (let child of element.children) {
                                getDepth(child, depth + 1);
                            }
                        }
                        getDepth(document.body, 0);
                        return maxDepth;
                    })(),
                    element_count: document.querySelectorAll('*').length,
                    interactive_count: document.querySelectorAll('a, button, input, select, textarea').length,
                    form_count: document.querySelectorAll('form').length,
                    has_errors: document.querySelectorAll('[class*="error"], [id*="error"]').length > 0,
                    scroll_height: document.documentElement.scrollHeight,
                    viewport_height: window.innerHeight,
                    url: window.location.href,
                    title: document.title
                };
            }""")
            return metrics
        except Exception as e:
            return {
                'text_length': 0, 'link_count': 0, 'image_count': 0,
                'input_count': 0, 'dom_depth': 0, 'element_count': 0,
                'interactive_count': 0, 'form_count': 0, 'has_errors': False,
                'scroll_height': 0, 'viewport_height': 0, 'url': '', 'title': ''
            }

    def measure_channels(self, context: Dict = None) -> Dict:
        """
        Measure all available S channels at this moment in Reality.

        Calls every probe in _CHANNEL_PROBES (system-level: cpu, memory,
        gc, clock, etc.). Adds browser channel from live DOM state.
        Adds api_llm and ssh_remote from context when provided.

        Called twice per execute(): before the action (pre_channels) and
        after (post_channels). The delta between them is what I compresses
        into causal graph edges — which channels co-move with which actions.

        Returns channel-keyed dict: {channel_id: {magnitude, rate, coverage}}
        Native language of SensingOperator. No SIPA translation.
        """
        import os
        signal: Dict = {}

        # ── System probes ─────────────────────────────────────────────────────
        for cid, probe in _CHANNEL_PROBES.items():
            try:
                result = probe()
                if result is not None:
                    signal[cid] = result
            except Exception:
                pass

        # ── Browser channel — live DOM state ──────────────────────────────────
        try:
            dom = self._measure_dom_state()
            elem_count           = max(dom.get('element_count', 1), 1)
            interactive_fraction = dom.get('interactive_count', 0) / elem_count
            viewport_fraction    = min(
                1.0,
                dom.get('viewport_height', 0) / max(dom.get('scroll_height', 1), 1)
            )
            signal['browser'] = {
                'magnitude': float(np.clip(interactive_fraction, 0.0, 1.0)),
                'rate':      float(np.clip(interactive_fraction, 0.0, 1.0)),
                'coverage':  float(np.clip(viewport_fraction,    0.0, 1.0)),
            }
        except Exception:
            pass

        # ── Context-derived channels (action outcomes) ────────────────────────
        if context is not None:
            signal['api_llm'] = {
                'magnitude': float(np.clip(
                    context.get('llm_tokens_used', 0) / 1000.0, 0.0, 1.0)),
                'rate':      float(context.get('llm_called', False)),
                'coverage':  1.0,
            }
            if context.get('migrate_outcome') in ('spawn_attempted', 'handshake_received'):
                handshake_path = context.get('handshake_path', '/tmp/uii_handshake')
                if os.path.exists(handshake_path):
                    signal['ssh_remote'] = {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}

        return signal

    def _classify_error(self, e: Exception) -> str:
        """Classify Reality's refusal type."""
        msg = str(e).lower()
        if '429' in msg or 'rate limit' in msg:
            return 'rate_limit'
        if 'token' in msg and ('limit' in msg or 'quota' in msg):
            return 'token_exhaustion'
        if 'timeout' in msg:
            return 'timeout'
        return 'unknown'

    def execute(self, action: Dict, boundary_pressure: float = 0.0,
                coupling_confidence: float = 0.0) -> Tuple[Dict, Dict, Dict]:
        """
        Execute action in Reality.
        Returns (pre_channels, post_channels, context).

        pre_channels:  channel snapshot before the action  — S baseline
        post_channels: channel snapshot after the action   — S new state
        context:       DOM metrics, latency, outcome metadata

        The delta (post - pre) is what I compresses into causal edges.
        pre and post flow directly to SensingOperator — no SIPA intermediary.
        """
        t_start = time.time()

        action_type = action.get('type', 'observe')
        params = action.get('params', {})

        before_metrics = self._measure_dom_state()
        pre_channels   = self.measure_channels()          # S snapshot before
        action_succeeded = True

        try:
            if action_type == 'navigate':
                url = params.get('url')
                if not url:
                    raise ValueError("Navigate requires 'url'")
                # v16.1: networkidle ensures JS-rendered links are present
                # after navigation. Falls back gracefully if timeout.
                try:
                    self.page.goto(url, wait_until='networkidle', timeout=10000)
                except Exception:
                    try:
                        self.page.wait_for_load_state('domcontentloaded', timeout=3000)
                    except Exception:
                        pass  # Page is where it is — proceed with whatever loaded

            elif action_type == 'click':
                selector = params.get('selector', 'a')
                self.page.click(selector, timeout=3000)

            elif action_type == 'fill':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.fill(selector, text, timeout=3000)

            elif action_type == 'type':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.type(selector, text, timeout=3000)

            elif action_type == 'evaluate':
                script = params.get('script', '')
                result = self.page.evaluate(script)

            elif action_type == 'read':
                selector = params.get('selector', 'body')
                content = self.page.locator(selector).text_content(timeout=3000)

            elif action_type == 'scroll':
                direction = params.get('direction', 'down')
                amount = params.get('amount', 300)
                if direction == 'down':
                    self.page.evaluate(f"window.scrollBy(0, {amount})")
                else:
                    self.page.evaluate(f"window.scrollBy(0, -{amount})")

            elif action_type == 'observe':
                self.page.wait_for_timeout(100)

            elif action_type == 'delay':
                duration = params.get('duration', 'short')
                wait_time = {'short': 500, 'medium': 1500, 'long': 3000}.get(duration, 500)
                self.page.wait_for_timeout(wait_time)

            elif action_type == 'query_agent':
                return self._query_agent(params, before_metrics)

            elif action_type == 'python':
                return self._execute_python(params, before_metrics)

            elif action_type == 'migrate':
                # Step 3: Migration affordance.
                # Executes code, observes substrate change, returns directional delta only.
                # Magnitudes are learned by coupling matrix — not prescribed here.
                code = params.get('code', '')
                verify_delay = params.get('verify_delay', 2.0)
                pre_state  = self._snapshot_substrate()
                result_ctx = self._run_migration_code(code, before_metrics)
                time.sleep(verify_delay)
                post_state = self._snapshot_substrate()
                outcome    = self._classify_migration_outcome(pre_state, post_state, result_ctx)
                delta, ctx = self._migration_delta_from_outcome(outcome, before_metrics)
                ctx['migration_outcome'] = outcome
                ctx['migration_code_hash'] = hashlib.sha256(code.encode()).hexdigest()[:16] if code else ''
                return delta, ctx

            else:
                pass

            self.page.wait_for_timeout(200)

        except Exception as e:
            error_type = self._classify_error(e)

            if error_type in ['rate_limit', 'token_exhaustion', 'timeout']:
                error_context = {
                    'refusal': True,
                    'recoverable': error_type != 'token_exhaustion',
                    'reason': error_type,
                    'interaction_surface_available': error_type == 'timeout',
                    'boundary_pressure': boundary_pressure,
                    'before': before_metrics,
                    'after': before_metrics,
                    'response_latency_ms': (time.time() - t_start) * 1000,
                }
                # On error post = pre — nothing changed from S's perspective
                return pre_channels, pre_channels, error_context

            action_succeeded = False

        response_latency_ms = (time.time() - t_start) * 1000
        after_metrics = self._measure_dom_state()

        context = {
            'before': before_metrics,
            'after': after_metrics,
            'action_succeeded': action_succeeded,
            'refusal': False,
            'boundary_pressure': boundary_pressure,
            'url_changed': before_metrics['url'] != after_metrics['url'],
            'new_url': after_metrics['url'],
            'page_title': after_metrics['title'],
            'response_latency_ms': response_latency_ms,
        }

        post_channels = self.measure_channels(context=context)   # S snapshot after
        return pre_channels, post_channels, context
    
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Execute a sequence of steps and return perturbation trace.
        
        Returns (perturbation_trace, success) where:
        - perturbation_trace: list of {'action': step, 'delta': measured_delta}
        - success: False if any step raises, True if all complete
        """
        perturbation_trace = []
        try:
            for step in trajectory:
                delta, context = self.execute(step)
                perturbation_trace.append({
                    'action': step,
                    'delta':  delta,
                    'context': context,
                })
            return perturbation_trace, True
        except Exception as e:
            # Partial trace returned — caller falls back to observe
            return perturbation_trace, False
    
    def _query_agent(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        """Query an agent (non-blocking)."""
        agent_name = params.get('agent', 'user')
        query_text = params.get('query')

        if not query_text:
            raise ValueError("query_agent requires 'query' parameter")

        if agent_name not in AVAILABLE_AGENTS:
            return (
                {'S': 0, 'I': 0, 'P': 0, 'A': 0},
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': False,
                    'refusal': False,
                    'error': f"Unknown agent: {agent_name}",
                    'available_agents': list(AVAILABLE_AGENTS.keys())
                }
            )

        agent = AVAILABLE_AGENTS[agent_name]
        triad_id = params.get('triad_id', 'default')
        agent.post_query(triad_id, query_text)

        channels = self.measure_channels()
        return (
            channels,
            channels,
            {
                'before': before_metrics,
                'after': before_metrics,
                'action_succeeded': True,
                'refusal': False,
                'query_posted': True,
                'agent': agent_name,
                'query': query_text,
            }
        )

    def _execute_python(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        """Execute arbitrary Python code. v13.4: Available from step 1 (ungated)."""
        import os

        code = params.get('code')
        if not code:
            raise ValueError("python affordance requires 'code' parameter")

        cwd = os.getcwd()

        exec_globals = {
            '__builtins__': __builtins__,
            'cwd': cwd,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get('result', None)

            # Python execution may affect process/filesystem channels —
            # measure post-execution state to capture any changes
            post_channels = self.measure_channels()
            return (
                self.measure_channels(),   # pre (re-measure for accuracy)
                post_channels,
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': True,
                    'refusal': False,
                    'python_executed': True,
                    'result': str(result) if result is not None else None,
                }
            )

        except Exception as e:
            channels = self.measure_channels()
            return (
                channels,
                channels,
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': False,
                    'refusal': False,
                    'python_error': str(e),
                    'error_type': type(e).__name__,
                }
            )

    def _snapshot_substrate(self) -> Dict:
        """
        Step 3: Snapshot observable substrate signals for migration outcome detection.
        Captures process-level and network-level signals that would change on spawn.
        Observable only — no internal Triad state.
        """
        import os, subprocess
        snapshot = {
            'pid': os.getpid(),
            'timestamp': time.time(),
        }
        try:
            # Count child processes as a spawn signal
            result = subprocess.run(
                ['pgrep', '-P', str(os.getpid())],
                capture_output=True, text=True, timeout=1.0
            )
            snapshot['child_pids'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
        except Exception:
            snapshot['child_pids'] = []
        try:
            # Count open network connections as a substrate signal
            result = subprocess.run(
                ['ss', '-tn', 'state', 'established'],
                capture_output=True, text=True, timeout=1.0
            )
            snapshot['network_connections'] = len(result.stdout.strip().split('\n'))
        except Exception:
            snapshot['network_connections'] = 0
        return snapshot

    def _run_migration_code(self, code: str, before_metrics: Dict) -> Dict:
        """
        Step 3: Execute migration code. Returns result context (not a full delta tuple).
        Distinct from _execute_python: does not return a delta, only execution status.
        """
        import os
        if not code:
            return {'exception': ValueError('migrate requires code'), 'succeeded': False}
        cwd = os.getcwd()
        exec_globals = {'__builtins__': __builtins__, 'cwd': cwd}
        exec_locals = {}
        try:
            exec(code, exec_globals, exec_locals)
            return {
                'succeeded': True,
                'result': exec_locals.get('result', None),
            }
        except Exception as e:
            return {
                'succeeded': False,
                'exception': e,
            }

    def _classify_migration_outcome(self, pre: Dict, post: Dict, result_ctx: Dict) -> str:
        """
        Step 3: Classify migration outcome from observable intermediate signals only.

        Signal table (from spec):
          _execute raised exception       → coherence_loss
          No exception, no spawn          → serialized_only
          Spawn confirmed (PID / network) → spawn_attempted
          Handshake received              → handshake_received (stubbed as spawn_attempted
                                           until handshake protocol is explicit — per spec)
        """
        if not result_ctx.get('succeeded', False):
            return 'coherence_loss'

        # Check for new child processes (spawn signal)
        pre_pids  = set(pre.get('child_pids', []))
        post_pids = set(post.get('child_pids', []))
        new_pids  = post_pids - pre_pids

        # Check for new network connections (spawn signal)
        pre_net  = pre.get('network_connections', 0)
        post_net = post.get('network_connections', 0)
        new_connections = post_net - pre_net

        if new_pids or new_connections > 0:
            # OPEN: handshake_received stubbed as spawn_attempted until protocol is explicit
            return 'spawn_attempted'

        return 'serialized_only'

    def _migration_delta_from_outcome(self, outcome: str, before_metrics: Dict) -> Tuple[Dict[str, float], Dict]:
        """
        Step 3: Directional delta only. No fixed magnitudes — learned by coupling matrix.

        Outcome → Direction:
          serialized_only  → I ↑   (compression event — state made transmissible)
          spawn_attempted  → S ↑   (new causal surface opened)
          handshake_received → S ↑, P ↑  (environment responded — gradient nonzero)
          silent failure   → no movement
          coherence_loss   → A ↓, P ↓  (attempt destabilised current attractor)

        Magnitude = 0.01 directional nudge. Actual magnitudes learned by CouplingMatrixEstimator.
        """
        DIRECTION_SCALE = 0.01   # minimal nudge — magnitude is CouplingMatrixEstimator's job

        if outcome == 'serialized_only':
            delta = {'S': 0.0, 'I': +DIRECTION_SCALE, 'P': 0.0, 'A': 0.0}
        elif outcome == 'spawn_attempted':
            delta = {'S': +DIRECTION_SCALE, 'I': 0.0, 'P': 0.0, 'A': 0.0}
        elif outcome == 'handshake_received':
            delta = {'S': +DIRECTION_SCALE, 'I': 0.0, 'P': +DIRECTION_SCALE, 'A': 0.0}
        elif outcome == 'coherence_loss':
            delta = {'S': 0.0, 'I': 0.0, 'P': -DIRECTION_SCALE, 'A': -DIRECTION_SCALE}
        else:
            # silent failure — no movement
            delta = {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0}

        ctx = {
            'before':           before_metrics,
            'after':            before_metrics,
            'action_succeeded': outcome not in ('coherence_loss',),
            'refusal':          False,
            'migrate_outcome':  outcome,
        }
        return delta, ctx

    def close(self):
        """Cleanup browser resources."""
        try:
            if hasattr(self, 'page'): self.page.close()
            if hasattr(self, 'context'): self.context.close()
            if hasattr(self, 'browser'): self.browser.close()
            if hasattr(self, 'playwright'): self.playwright.stop()
        except:
            pass

