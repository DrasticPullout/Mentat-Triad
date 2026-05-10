#!/usr/bin/env python3
"""cli_observe.py — UII v19.0 — JSONL trajectory observer.

Tail a substrate JSONL stream and render trajectory + math-spine shape
analysis. Reads JSONL from a file (with --follow for live tailing) or
from stdin.

Optional --perturbations FILE watches a sidecar file for perturbation
events (one per line: 'TIMESTAMP\\tLABEL'). The forward transducer
appends a line each time it sends user input, the observer correlates
to the JSONL trajectory.

Usage:
    # tail a file once and exit when the file is fully consumed
    python cli_observe.py path/to/triad.jsonl

    # follow a file as the substrate writes to it
    python cli_observe.py --follow path/to/triad.jsonl

    # follow file with perturbation events from sidecar
    python cli_observe.py --follow path/to/triad.jsonl --perturbations /tmp/perts.log

    # read from stdin (for piping: substrate | cli_observe.py)
    cat path/to/triad.jsonl | python cli_observe.py -

This utility imports only `observability.py` — no substrate access.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterator, Optional, TextIO

from observability import ObservabilityEngine, JSONLRecord


def tail_lines(path: str, follow: bool, sleep_s: float = 0.05) -> Iterator[str]:
    """Yield lines from a file. With follow=True, keep watching for
    new lines after EOF (poll-based)."""
    with open(path, 'r') as f:
        # Start at end if follow mode AND file already has content?
        # No — start at beginning; user often wants full history. If
        # they want tail-only, use shell tools.
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                if not follow:
                    return
                time.sleep(sleep_s)


def stdin_lines() -> Iterator[str]:
    for line in sys.stdin:
        yield line


def read_perturbations_once(path: str, engine: ObservabilityEngine) -> int:
    """Read all existing perturbation entries from the sidecar synchronously.
    Returns the file position at end-of-file (for follow mode to continue from).
    Each line: 'TIMESTAMP\\tLABEL'."""
    pos = 0
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r') as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    ts_s, label = raw.rstrip('\n').split('\t', 1)
                    ts = float(ts_s)
                except (ValueError, IndexError):
                    continue
                engine.mark_perturbation(label=label, timestamp=ts)
            pos = f.tell()
    except Exception:
        pass
    return pos


def watch_perturbations(path: str, engine: ObservabilityEngine,
                        start_pos: int = 0,
                        sleep_s: float = 0.05) -> None:
    """Background loop: watch the sidecar file for new entries appended
    after start_pos. Used in follow mode."""
    last_pos = start_pos
    while True:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    f.seek(last_pos)
                    for raw in f:
                        if not raw.strip():
                            continue
                        try:
                            ts_s, label = raw.rstrip('\n').split('\t', 1)
                            ts = float(ts_s)
                        except (ValueError, IndexError):
                            continue
                        engine.mark_perturbation(label=label, timestamp=ts)
                    last_pos = f.tell()
        except Exception:
            pass
        time.sleep(sleep_s)


def render_record_line(record: JSONLRecord, engine: ObservabilityEngine) -> str:
    """One-line live display of a record, with running baseline stats."""
    commit_str = ''
    if record.commit:
        ctype = record.commit.get('type', '?')
        commit_str = f"  commit={ctype}"
    return (f"iter={record.iter:6d}  "
            f"t={record.t:.3f}  "
            f"δf_rel={record.delta_f_rel:+.5f}  "
            f"baseline μ={engine.baseline.mean:+.5f} "
            f"σ={engine.baseline.std:.5f}"
            f"{commit_str}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Tail a substrate JSONL stream and render '
                    'trajectory + shape analysis.',
    )
    parser.add_argument('source', nargs='?', default='-',
                        help='Path to JSONL file, or "-" for stdin (default: -)')
    parser.add_argument('--follow', '-f', action='store_true',
                        help='Keep watching the file for new lines after EOF')
    parser.add_argument('--perturbations', '-p', default=None,
                        help='Sidecar file of perturbation events to watch '
                             '(format per line: "TIMESTAMP\\tLABEL")')
    parser.add_argument('--baseline-window', type=int, default=30,
                        help='Rolling baseline window in iterations (default: 30)')
    parser.add_argument('--metabolize-horizon', type=int, default=5,
                        help='Iterations to observe after a perturbation '
                             '(default: 5)')
    parser.add_argument('--quiet-trajectory', '-q', action='store_true',
                        help='Suppress per-iteration trajectory lines; '
                             'show only perturbation analyses')
    args = parser.parse_args()

    engine = ObservabilityEngine(
        baseline_window    = args.baseline_window,
        metabolize_horizon = args.metabolize_horizon,
    )

    # Optional sidecar perturbation events
    if args.perturbations:
        # Read existing events synchronously so they're registered
        # before JSONL processing begins. Critical for one-shot mode
        # where the JSONL file is fully consumed before any background
        # thread would have a chance to poll.
        end_pos = read_perturbations_once(args.perturbations, engine)
        n_loaded = len(engine._pending_events)
        print(f"# loaded {n_loaded} perturbation event(s) from "
              f"{args.perturbations}", file=sys.stderr)

        # In follow mode, also watch for newly-appended events
        if args.follow:
            import threading
            threading.Thread(
                target=watch_perturbations,
                args=(args.perturbations, engine, end_pos),
                daemon=True,
            ).start()

    # Source selection
    if args.source == '-':
        line_source = stdin_lines()
        print("# reading JSONL from stdin", file=sys.stderr)
    else:
        line_source = tail_lines(args.source, follow=args.follow)
        mode = 'follow' if args.follow else 'one-shot'
        print(f"# reading JSONL from {args.source} ({mode})", file=sys.stderr)

    print(f"# baseline window={args.baseline_window}  "
          f"metabolize horizon={args.metabolize_horizon}",
          file=sys.stderr)
    print("# trajectory: each line is one substrate iteration",
          file=sys.stderr)
    print("", file=sys.stderr)

    seen_completed_events: set = set()
    try:
        for record in engine.stream(line_source):
            if not args.quiet_trajectory:
                print(render_record_line(record, engine))

            # When an event finishes its horizon, render its analysis once
            for i, event in enumerate(engine.events):
                key = (event.timestamp, event.label)
                if key not in seen_completed_events:
                    seen_completed_events.add(key)
                    print()
                    print(event.render_text())
                    print()
    except KeyboardInterrupt:
        pass

    # Render any pending events with partial trajectories
    pending = [e for e in engine._pending_events if e.observed_trajectory]
    if pending:
        print()
        print("# pending events (partial trajectory observed):", file=sys.stderr)
        for event in pending:
            print()
            print(event.render_text())
            print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
