"""Helpers for Lesson 2 — Voice in your App.

Wraps three things the notebook needs:
  1. mint_token() — POST to VB's /api/v1/token with your API key
  2. vb()         — shell out to the `vb` CLI and return parsed output
  3. voice_widget() — render an in-notebook voice widget using the
                      Vocal Bridge React SDK loaded from a CDN

Voice runs in the rendered widget (browser context). Python only handles
auth, agent config, and inspection of call logs.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import requests
from dotenv import load_dotenv, find_dotenv


VB_API_URL = "https://vocalbridgeai.com"

# Pinned versions for CDN imports. Bump when newer VB SDKs ship.
VB_REACT_VER = "0.1.1"
VB_SDK_VER = "0.1.1"
REACT_VER = "18"

def load_env():
    load_dotenv(find_dotenv())



@dataclass
class TokenData:
    """Voice session token returned by VB's /api/v1/token endpoint.

    Field names use VB-native vocabulary; the underlying transport details
    are an implementation detail learners don't need to think about.
    """
    connection_url: str       # the wss:// endpoint the SDK connects to
    token: str                # short-lived JWT
    session_name: str         # unique name for this voice session
    agent_mode: str           # e.g. "openai_concierge"

    def as_dict(self) -> dict[str, Any]:
        return {
            "connection_url": self.connection_url,
            "token": self.token,
            "session_name": self.session_name,
            "agent_mode": self.agent_mode,
        }


def mint_token(
    agent_id: str,
    api_key: str | None = None,
    participant_name: str = "Notebook Learner",
) -> TokenData:
    """POST /api/v1/token. Returns a VB voice session token for the widget."""
    api_key = api_key or os.environ["VOCAL_BRIDGE_API_KEY"]
    res = requests.post(
        f"{VB_API_URL}/api/v1/token",
        headers={
            "X-API-Key": api_key,
            "X-Agent-Id": agent_id,
            "Content-Type": "application/json",
        },
        json={"participant_name": participant_name},
        timeout=15,
    )
    res.raise_for_status()
    data = res.json()
    # API returns transport-level field names; alias them to the
    # neutral, transport-agnostic names exposed by the SDK.
    return TokenData(
        connection_url=data.get("connection_url") or data["livekit_url"],
        token=data["token"],
        session_name=data["room_name"],
        agent_mode=data.get("agent_mode", ""),
    )


def vb(*args: str, json_output: bool = False) -> Any:
    """Run `vb <args>` and return stdout. Set json_output=True to parse JSON."""
    cmd = ["vb", *args]
    if json_output and "--json" not in args:
        cmd.append("--json")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"vb {args} failed:\n{proc.stderr or proc.stdout}")
    out = proc.stdout
    if json_output:
        # The CLI prints a human-readable table first, then a JSON block
        # separated by `--- JSON ---`. Anchor on that marker when present.
        marker = "--- JSON ---"
        if marker in out:
            out = out.split(marker, 1)[1]
        for i, ch in enumerate(out):
            if ch in "{[":
                return json.loads(out[i:])
        raise ValueError(f"no JSON found in output:\n{out}")
    return out


def append_to_env(key: str, value: str, env_path: str | Path = ".env") -> None:
    """Append or update a key=value line in .env."""
    p = Path(env_path)
    lines = p.read_text().splitlines() if p.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    p.write_text("\n".join(lines) + "\n")
    os.environ[key] = value


# ── In-notebook voice widget (Vocal Bridge React SDK) ─────────────────
_WIDGET_TEMPLATE = Template(r"""
<div id="$root_id" style="font-family: -apple-system, system-ui, sans-serif; max-width: 720px; min-height: ${height}px; padding: 16px; border: 1px solid #ddd; border-radius: 12px; color: #888;">loading Vocal Bridge React SDK…</div>
<script type="module">
import React, { useState, useEffect, useCallback, useRef } from 'https://esm.sh/react@$react_ver';
import ReactDOM from 'https://esm.sh/react-dom@$react_ver/client';
import {
  VocalBridgeProvider, useVocalBridge, useTranscript, useAgentActions,
} from 'https://esm.sh/@vocalbridgeai/react@$vb_react_ver?deps=react@$react_ver,react-dom@$react_ver';
import { ConnectionState } from 'https://esm.sh/@vocalbridgeai/sdk@$vb_sdk_ver';

const e = React.createElement;
const TOKEN = $token_payload;

// Pre-minted token; the React SDK consumes it via a custom tokenProvider
// so no API key ever reaches the browser. The SDK's contract uses the
// underlying transport field names; we map our VB-native names to those.
const tokenProvider = async () => ({
  url: TOKEN.connection_url,
  token: TOKEN.token,
  room_name: TOKEN.session_name,
  participant_identity: 'notebook-learner',
  expires_in: 3600,
  agent_mode: TOKEN.agent_mode,
});

// ── Tic-tac-toe pure helpers ─────────────────────────────────────────
const WIN_LINES = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6],
];

function detectWin(cells) {
  for (const [a,b,c] of WIN_LINES) {
    if (cells[a] && cells[a] === cells[b] && cells[b] === cells[c]) {
      return { winner: cells[a], line: [a,b,c] };
    }
  }
  return { winner: null, line: null };
}

function gameStatus(cells, userSym, agentSym) {
  const { winner } = detectWin(cells);
  if (winner === userSym) return 'user_wins';
  if (winner === agentSym) return 'agent_wins';
  if (cells.every((c) => c)) return 'draw';
  return 'playing';
}

function statusBanner(status, userSym, agentSym) {
  if (status === 'user_wins')  return ['🏆  You won! ('+userSym+')',     '#10b981'];
  if (status === 'agent_wins') return ['🤖  Agent wins ('+agentSym+')',  '#dc2626'];
  if (status === 'draw')       return ['🤝  Draw',                        '#666'];
  return [null, null];
}

// ── Styles ───────────────────────────────────────────────────────────
const ROLE_COLOR = { user: '#4f46e5', agent: '#10b981', app: '#f59e0b' };
const styles = {
  row:    { display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 },
  btn: (kind) => ({
    padding: '10px 18px', borderRadius: 8, border: 'none',
    background: kind === 'danger' ? '#ef4444' : '#4f46e5',
    color: 'white', fontWeight: 600, cursor: 'pointer',
  }),
  state:  { color: '#666', fontFamily: 'monospace', fontSize: 13 },
  board:  { display: 'grid', gridTemplateColumns: 'repeat(3, 64px)', gap: 6, margin: '14px 0' },
  cell: (filled, winning) => ({
    height: 64, fontSize: 28, fontWeight: 600,
    background: winning ? '#dcfce7' : 'white',
    color: filled ? '#1a1133' : '#bbb',
    border: '2px solid ' + (winning ? '#10b981' : '#ddd'),
    borderRadius: 6,
    cursor: filled ? 'default' : 'pointer',
  }),
  banner: (color) => ({
    padding: '8px 14px', borderRadius: 8, marginBottom: 10,
    background: color + '14', color: color, fontWeight: 600, fontSize: 14,
  }),
  transcript: { height: 220, overflowY: 'auto', padding: 12, background: '#fafafa', borderRadius: 8, fontSize: 14, lineHeight: 1.5 },
  roleTag: (role) => ({
    color: ROLE_COLOR[role] || '#999', fontWeight: 600,
    fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', marginRight: 6,
  }),
  err: { color: '#dc2626', fontSize: 13, marginTop: 6 },
};

// ── Tic-tac-toe widget ───────────────────────────────────────────────
function VoiceUI() {
  const { state, connect, disconnect, error } = useVocalBridge();
  const { transcript } = useTranscript();
  const { onAction, sendAction } = useAgentActions();

  const [boardVisible, setBoardVisible] = useState(false);
  const [cells, setCells] = useState(Array(9).fill(''));
  const [userSym, setUserSym] = useState('X');
  const [status, setStatus] = useState('idle');

  // Refs hold the latest values so action handlers can read fresh state
  // without re-subscribing on every render.
  const cellsRef = useRef(cells);
  const userSymRef = useRef(userSym);
  useEffect(() => { cellsRef.current = cells; }, [cells]);
  useEffect(() => { userSymRef.current = userSym; }, [userSym]);

  const agentSym = userSym === 'X' ? 'O' : 'X';

  // Apply a mark and broadcast the new board state to the agent.
  const applyMark = useCallback((index, mark, syncKind) => {
    const prev = cellsRef.current;
    if (prev[index]) return null;  // cell already taken — ignore
    const next = [...prev];
    next[index] = mark;
    cellsRef.current = next;
    setCells(next);

    const us = userSymRef.current;
    const as = us === 'X' ? 'O' : 'X';
    const newStatus = gameStatus(next, us, as);
    setStatus(newStatus);

    const turn = mark === us ? 'agent' : 'user';
    const payload = {
      board: next,
      status: newStatus,
      turn,
      moveCount: next.filter((c) => c).length,
      userSymbol: us,
      agentSymbol: as,
    };
    if (syncKind === 'board_sync') {
      sendAction('board_sync', payload);
    } else if (syncKind === 'user_placed_mark') {
      sendAction('user_placed_mark', { ...payload, position: index, row: Math.floor(index / 3), col: index % 3, mark });
    }
    return next;
  }, [sendAction]);

  // ── Subscribe to incoming actions from the agent ───────────────────
  useEffect(() => {
    const offs = [
      onAction('show_tic_tac_toe', (p) => {
        const us = (p && p.userSymbol) || 'X';
        cellsRef.current = Array(9).fill('');
        userSymRef.current = us;
        setUserSym(us);
        setCells(Array(9).fill(''));
        setStatus('playing');
        setBoardVisible(true);
        // Send an initial baseline so the agent has authoritative state.
        sendAction('board_sync', {
          board: Array(9).fill(''),
          status: 'playing',
          turn: (p && p.firstTurn) || 'user',
          moveCount: 0,
          userSymbol: us,
          agentSymbol: us === 'X' ? 'O' : 'X',
        });
      }),
      onAction('place_mark', (p) => {
        if (!p || p.row === undefined || p.col === undefined) return;
        const as = userSymRef.current === 'X' ? 'O' : 'X';
        applyMark(p.row * 3 + p.col, as, 'board_sync');
      }),
      onAction('user_move', (p) => {
        if (!p || p.row === undefined || p.col === undefined) return;
        applyMark(p.row * 3 + p.col, userSymRef.current, 'board_sync');
      }),
      onAction('clear_ui', () => {
        cellsRef.current = Array(9).fill('');
        setBoardVisible(false);
        setCells(Array(9).fill(''));
        setStatus('idle');
      }),
    ];
    return () => offs.forEach((off) => off && off());
  }, [onAction, sendAction, applyMark]);

  // ── Click handler — places user mark, fires user_placed_mark ──────
  const handleCellClick = useCallback((i) => {
    if (status !== 'playing') return;
    if (cellsRef.current[i]) return;
    applyMark(i, userSymRef.current, 'user_placed_mark');
  }, [applyMark, status]);

  const isDisconnected = state === ConnectionState.Disconnected;
  const busy = state === ConnectionState.Connecting || state === ConnectionState.WaitingForAgent;

  const winInfo = detectWin(cells);
  const [bannerText, bannerColor] = statusBanner(status, userSym, agentSym);

  return e('div', null,
    e('div', { style: styles.row },
      e('button', {
        style: styles.btn(isDisconnected ? 'primary' : 'danger'),
        onClick: isDisconnected ? connect : disconnect,
        disabled: busy,
      }, isDisconnected ? 'Connect' : 'Disconnect'),
      e('span', { style: styles.state }, state),
    ),
    error && e('div', { style: styles.err }, error.message),
    bannerText && e('div', { style: styles.banner(bannerColor) }, bannerText),
    boardVisible && e('div', { style: styles.board },
      cells.map((c, i) => e('button', {
        key: i,
        style: styles.cell(!!c, winInfo.line && winInfo.line.includes(i)),
        onClick: () => handleCellClick(i),
      }, c)),
    ),
    e('div', { style: styles.transcript },
      transcript.length === 0
        ? e('div', { style: { color: '#aaa', fontStyle: 'italic' } }, 'transcript will appear here…')
        : transcript.map((t, i) => e('div', { key: i },
            e('span', { style: styles.roleTag(t.role) }, t.role),
            t.text,
          ))
    ),
  );
}

function App() {
  return e(VocalBridgeProvider,
    { options: { auth: { tokenProvider }, debug: false } },
    e(VoiceUI),
  );
}

const container = document.getElementById('$root_id');
container.style.color = '#222';
container.innerHTML = '';
ReactDOM.createRoot(container).render(e(App));
</script>
""")


def voice_widget(token: TokenData, height: int = 540) -> str:
    """Render an in-notebook voice widget using @vocalbridgeai/react.

    Pre-mints a token in Python; the React SDK consumes it via a custom
    tokenProvider so no API key ever reaches the browser. The widget
    handles tic-tac-toe state locally and syncs every move to the agent
    via `board_sync` (silent) and `user_placed_mark` (response-triggering).
    """
    return _WIDGET_TEMPLATE.substitute(
        root_id=f"vb-root-{int(time.time() * 1000)}",
        height=height,
        react_ver=REACT_VER,
        vb_react_ver=VB_REACT_VER,
        vb_sdk_ver=VB_SDK_VER,
        token_payload=json.dumps(token.as_dict()),
    )
