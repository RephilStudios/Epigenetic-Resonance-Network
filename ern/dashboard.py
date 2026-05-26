HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>ERN // DGX NEURAL CONSOLE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #050807;
  --bg2:       #080d0b;
  --bg3:       #0c1410;
  --border:    #1a2e22;
  --border2:   #0f1f17;
  --g0:        #00ff88;
  --g1:        #00cc66;
  --g2:        #008844;
  --g3:        #004422;
  --amber:     #ffaa00;
  --red:       #ff3a3a;
  --blue:      #00aaff;
  --dim:       #2a4a38;
  --text:      #b0d4c0;
  --text-dim:  #4a7a5a;
  --font-mono: 'Share Tech Mono', monospace;
  --font-hud:  'Orbitron', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-mono);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  height: 100dvh; 
  overflow: hidden;
  display: grid;
  grid-template-rows: 48px 1fr;
  grid-template-columns: 1fr 340px;
  grid-template-areas:
    "topbar topbar"
    "chat   sidebar";
  background-image:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px);
}

body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background: linear-gradient(transparent 50%, rgba(0,0,0,0.06) 50%);
  background-size: 100% 4px;
  animation: flicker 8s infinite;
}
@keyframes flicker {
  0%,97%,100%  { opacity:1 }
  98%          { opacity:0.92 }
  99%          { opacity:1 }
  99.5%        { opacity:0.88 }
}

#topbar {
  grid-area: topbar;
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px; gap: 24px;
  position: relative; overflow: hidden;
}
#topbar::-webkit-scrollbar { display: none; }
#topbar { -ms-overflow-style: none; scrollbar-width: none; }
#topbar::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg, transparent, var(--g0), transparent);
  animation: sweep 4s linear infinite;
}
@keyframes sweep { from{transform:translateX(-100%)} to{transform:translateX(100%)} }

.brand {
  font-family: var(--font-hud); font-size: 0.8rem; font-weight: 900;
  color: var(--g0); letter-spacing: 0.15em; white-space: nowrap;
  text-shadow: 0 0 12px rgba(0,255,136,0.6);
}
.brand span { color: var(--text-dim); font-weight:400; }

#status-ticker {
  flex: 1; font-size: 0.7rem; color: var(--g2); letter-spacing: 0.05em;
  overflow: hidden; white-space: nowrap; position: relative;
}
#ticker-inner {
  display: inline-block; animation: ticker-scroll 0s linear infinite; padding-left: 100%;
}
@keyframes ticker-scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }

.hud-pill {
  font-family: var(--font-hud); font-size: 0.6rem; padding: 3px 10px;
  border: 1px solid var(--border); border-radius: 2px;
  color: var(--text-dim); white-space: nowrap; letter-spacing: 0.08em;
}
.hud-pill.live { border-color: var(--g3); color: var(--g1); }
.hud-pill.live::before { content:'● '; animation: blink 1.2s step-end infinite; }
@keyframes blink { 50%{opacity:0} }
#node-count { color: var(--g0); font-weight:bold; }

#chat-area { grid-area: chat; display: flex; flex-direction: column; border-right: 1px solid var(--border2); overflow: hidden; }

#activity-bar {
  padding: 6px 16px; background: var(--bg2); border-bottom: 1px solid var(--border2);
  font-size: 0.7rem; color: var(--text-dim); display: flex; align-items: center; gap: 16px; min-height: 28px;
}
#activity-bar::-webkit-scrollbar { display: none; }
#activity-bar { -ms-overflow-style: none; scrollbar-width: none; }
#activity-bar .phase { display: flex; align-items: center; gap: 6px; }
.phase-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--dim); transition: background 0.2s, box-shadow 0.2s; }
.phase-dot.active { background: var(--g0); box-shadow: 0 0 8px var(--g0); animation: pulse-dot 0.8s ease-in-out infinite alternate; }
.phase-dot.done { background: var(--g2); }
.phase-dot.error { background: var(--red); box-shadow: 0 0 6px var(--red); }
@keyframes pulse-dot { from{opacity:1} to{opacity:0.4} }
#activity-text { flex:1; }

#chatBox { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; scroll-behavior: smooth; }
#chatBox::-webkit-scrollbar { width: 4px; }
#chatBox::-webkit-scrollbar-track { background: transparent; }
#chatBox::-webkit-scrollbar-thumb { background: var(--g3); border-radius: 2px; }

.msg { max-width: 82%; padding: 10px 14px; font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; animation: msg-in 0.18s ease; position: relative; }
@keyframes msg-in { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

.msg.user { align-self: flex-end; background: var(--bg3); border: 1px solid var(--g3); border-right: 3px solid var(--g0); color: var(--text); }
.msg.user::before { content: 'USR ▶'; display: block; font-family: var(--font-hud); font-size: 0.58rem; color: var(--g2); margin-bottom: 5px; letter-spacing: 0.1em; }

.msg.bot { align-self: flex-start; background: var(--bg2); border: 1px solid var(--border); border-left: 3px solid var(--g2); color: var(--text); }
.msg.bot::before { content: 'ERN ◀'; display: block; font-family: var(--font-hud); font-size: 0.58rem; color: var(--g1); margin-bottom: 5px; letter-spacing: 0.1em; text-shadow: 0 0 8px rgba(0,255,136,0.5); }
.msg.bot.thinking { border-left-color: var(--amber); color: var(--text-dim); }
.msg.bot.thinking::before { content: 'SYS ◀'; color: var(--amber); }

.thinking-dots { display:inline-block; }
.thinking-dots::after { content: '...'; animation: dots 1.2s steps(4, end) infinite; }
@keyframes dots { 0% { content: '.  '; } 33% { content: '.. '; } 66% { content: '...'; } 100%{ content: '.  '; } }

.cognition-monitor { margin-top: 8px; background: rgba(0, 0, 0, 0.4); border: 1px solid var(--border2); border-left: 2px solid var(--g2); font-size: 0.72rem; font-family: var(--font-mono); }
.cognition-header { padding: 6px 10px; background: rgba(0, 255, 136, 0.04); cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-family: var(--font-hud); font-size: 0.6rem; letter-spacing: 0.08em; color: var(--g0); user-select: none; transition: background 0.15s; }
.cognition-header:hover { background: rgba(0, 255, 136, 0.08); }
.cognition-header::after { content: '▼'; font-size: 0.55rem; transition: transform 0.2s; color: var(--g2); }
.cognition-monitor.open .cognition-header::after { transform: rotate(-180deg); }
.cognition-body { display: none; padding: 8px 12px; border-top: 1px solid var(--border2); animation: slide-down 0.2s ease-out; }
.cognition-monitor.open .cognition-body { display: block; }
@keyframes slide-down { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.cognition-step { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.cognition-step:last-child { margin-bottom: 0; }
.step-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.step-dot.active { background: var(--g0); box-shadow: 0 0 6px var(--g0); animation: pulse-dot 1.5s infinite; }
.step-dot.inactive { background: var(--text-dim); }
.step-dot.disabled { background: var(--dim); }
.step-dot.queued { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
@keyframes pulse-dot { 0% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.3); opacity: 1; } 100% { transform: scale(1); opacity: 0.8; } }
.step-title { font-weight: bold; color: var(--text); margin-right: 4px; }
.step-detail { color: var(--text-dim); }

#input-area { padding: 12px 16px; background: var(--bg3); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }
#userInput { flex: 1; padding: 10px 14px; background: var(--bg); border: 1px solid var(--border); border-bottom: 2px solid var(--g3); color: var(--g0); font-family: var(--font-mono); font-size: 0.88rem; outline: none; caret-color: var(--g0); transition: border-color 0.2s; }
#userInput:focus { border-color: var(--g2); border-bottom-color: var(--g0); box-shadow: 0 0 0 1px rgba(0,255,136,0.1); }
#userInput::placeholder { color: var(--text-dim); }
#sendBtn { padding: 10px 20px; background: transparent; border: 1px solid var(--g2); color: var(--g0); font-family: var(--font-hud); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; cursor: pointer; transition: all 0.15s; position: relative; overflow: hidden; }
#sendBtn:hover:not(:disabled) { background: var(--g3); border-color: var(--g0); box-shadow: 0 0 16px rgba(0,255,136,0.25); }
#sendBtn:disabled { opacity: 0.3; cursor: not-allowed; }
#sendBtn::after { content:''; position:absolute; inset:0; background: linear-gradient(90deg, transparent, rgba(0,255,136,0.15), transparent); transform: translateX(-100%); transition: transform 0.3s; }
#sendBtn:hover:not(:disabled)::after { transform: translateX(100%); }

#controls-strip { padding: 8px 16px; background: var(--bg2); border-top: 1px solid var(--border2); display: flex; gap: 20px; align-items: center; font-size: 0.72rem; color: var(--text-dim); flex-wrap: wrap; }
.ctrl-group { display:flex; align-items:center; gap:8px; }
.ctrl-label { font-family: var(--font-hud); font-size:0.6rem; letter-spacing:0.1em; }
input[type="range"] { -webkit-appearance: none; width: 90px; height: 3px; background: var(--dim); outline: none; cursor: pointer; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; background: var(--g0); box-shadow: 0 0 6px var(--g0); cursor: pointer; }
.val-badge { font-family: var(--font-hud); font-size: 0.65rem; color: var(--g0); min-width: 28px; }
select { background: var(--bg); color: var(--g1); border: 1px solid var(--border); padding: 4px 8px; font-family: var(--font-mono); font-size: 0.75rem; outline: none; cursor: pointer; max-width: 180px; }
select option { background: var(--bg); }

#sidebar { grid-area: sidebar; display: flex; flex-direction: column; background: var(--bg2); overflow: hidden; }
.panel-tabs { display: flex; border-bottom: 1px solid var(--border); }
.tab { flex: 1; padding: 9px 4px; text-align: center; cursor: pointer; font-family: var(--font-hud); font-size: 0.58rem; letter-spacing: 0.1em; color: var(--text-dim); border-bottom: 2px solid transparent; transition: all 0.15s; }
.tab.active { color: var(--g0); border-bottom-color: var(--g0); background: var(--bg3); }
.tab:hover:not(.active) { color: var(--g2); }
.panel-body { flex:1; overflow:hidden; display:flex; flex-direction:column; }
.panel-section { display:none; flex:1; flex-direction:column; overflow:hidden; }
.panel-section.visible { display:flex; }
#neuro-canvas { width:100%; height:70px; display:block; background: var(--bg); border-bottom:1px solid var(--border2); }

#memScroll { flex:1; overflow-y:auto; padding:10px; }
#memScroll::-webkit-scrollbar { width:3px; }
#memScroll::-webkit-scrollbar-thumb { background: var(--g3); }
.memory-card { border-left: 2px solid var(--g3); padding: 8px 100px 8px 10px; margin-bottom: 8px; font-size: 0.76rem; color: var(--text); background: var(--bg3); position: relative; animation: card-in 0.2s ease; transition: border-color 0.2s; }
.memory-card:hover { border-left-color: var(--g0); }
@keyframes card-in { from{opacity:0;transform:translateX(6px)} to{opacity:1;transform:none} }
.memory-card .mc-tags { font-size: 0.65rem; color: var(--g2); font-family: var(--font-hud); letter-spacing: 0.06em; margin-bottom: 3px; }
.memory-card .mc-resonance { position:absolute; top:6px; right:8px; font-family: var(--font-hud); font-size: 0.6rem; color: var(--amber); }
.memory-card .mc-energy { position:absolute; top:18px; right:8px; font-family: var(--font-hud); font-size: 0.55rem; color: var(--blue); }
.memory-card .mc-forget { position:absolute; bottom:6px; right:8px; font-family: var(--font-hud); font-size: 0.55rem; color: var(--text-dim); cursor: pointer; border: none; background: none; letter-spacing: 0.08em; transition: color 0.15s, text-shadow 0.15s; outline: none; }
.memory-card .mc-forget:hover { color: var(--red); text-shadow: 0 0 6px var(--red); }
.no-mem { color: var(--text-dim); font-size:0.78rem; padding:12px; font-style:italic; }

#vaultSearch { width: calc(100% - 16px); margin: 8px; padding: 8px 12px; background: var(--bg); border: 1px solid var(--border); border-bottom: 2px solid var(--dim); color: var(--g0); font-family: var(--font-mono); font-size: 0.8rem; outline: none; caret-color: var(--g0); transition: border-color 0.2s, box-shadow 0.2s; }
#vaultSearch:focus { border-color: var(--g2); border-bottom-color: var(--g0); box-shadow: 0 0 8px rgba(0,255,136,0.1); }
#vaultScroll { flex:1; overflow-y:auto; padding:10px; }
#vaultScroll::-webkit-scrollbar { width:3px; }
#vaultScroll::-webkit-scrollbar-thumb { background: var(--g3); }

#deltaScroll { flex:1; overflow-y:auto; padding:8px; }
#deltaScroll::-webkit-scrollbar { width:3px; }
#deltaScroll::-webkit-scrollbar-thumb { background: var(--g3); }
#deltaStatsBar { padding: 6px 10px; font-size: 0.68rem; color: var(--text-dim); background: var(--bg3); border-bottom: 1px solid var(--border2); display: flex; gap: 12px; flex-wrap: wrap; }
.stat-pill { display:flex; gap:4px; align-items:center; }
.stat-pill .sp-val { font-family:var(--font-hud); font-size:0.65rem; }
.sp-ENCODE { color:var(--g0); } .sp-DECAY  { color:var(--amber); } .sp-BOOST  { color:var(--blue); } .sp-SLEEP  { color:var(--red); }
.delta-entry { display: flex; gap: 8px; align-items: flex-start; padding: 5px 6px; margin-bottom: 4px; border-left: 2px solid var(--border); font-size: 0.72rem; background: var(--bg3); animation: card-in 0.15s ease; }
.delta-entry.ENCODE { border-color: var(--g0); } .delta-entry.DECAY  { border-color: var(--amber); } .delta-entry.BOOST  { border-color: var(--blue); } .delta-entry.SLEEP  { border-color: var(--red); } .delta-entry.RESTORE{ border-color: #444; }
.de-op { font-family: var(--font-hud); font-size: 0.6rem; letter-spacing:0.08em; width: 52px; flex-shrink:0; padding-top:1px; }
.ENCODE .de-op { color:var(--g0); } .DECAY  .de-op { color:var(--amber); } .BOOST  .de-op { color:var(--blue); } .SLEEP  .de-op { color:var(--red); } .RESTORE .de-op { color:#555; }
.de-body { color: var(--text-dim); line-height:1.4; }
.de-size { font-family:var(--font-hud); font-size:0.58rem; color:#3a5a4a; margin-left:auto; white-space:nowrap; }

#rollback-row { padding: 8px 10px; border-top: 1px solid var(--border); background: var(--bg3); display: flex; gap: 8px; align-items: center; }
#rollback-row .ctrl-label { color: var(--text-dim); }
#rollbackN { width: 48px; padding: 4px 6px; background: var(--bg); border: 1px solid var(--border); color: var(--g1); font-family: var(--font-mono); font-size: 0.8rem; text-align: center; outline: none; }
#rollbackBtn { padding: 5px 12px; background: transparent; border: 1px solid var(--red); color: var(--red); font-family: var(--font-hud); font-size: 0.6rem; letter-spacing: 0.1em; cursor: pointer; transition: all 0.15s; }
#rollbackBtn:hover { background: rgba(255,58,58,0.15); box-shadow: 0 0 10px rgba(255,58,58,0.3); }

#statsGrid { flex:1; overflow-y:auto; padding:12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-content: start; }
.stat-card { background: var(--bg3); border: 1px solid var(--border); padding: 10px 12px; }
.stat-card .sc-label { font-family: var(--font-hud); font-size: 0.56rem; letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 4px; }
.stat-card .sc-val { font-family: var(--font-hud); font-size: 1.2rem; color: var(--g0); text-shadow: 0 0 10px rgba(0,255,136,0.4); }
.stat-card.wide { grid-column: span 2; }
.energy-bar-wrap { margin-top:8px; }
.energy-bar-label { font-size:0.62rem; color:var(--text-dim); margin-bottom:3px; display:flex; justify-content:space-between; }
.energy-bar { height:4px; background:var(--dim); position:relative; }
.energy-bar-fill { height:100%; background: linear-gradient(90deg, var(--g3), var(--g0)); transition: width 0.6s ease; }

#sleepBtn { margin: 10px; padding: 8px; background: transparent; border: 1px solid var(--red); color: var(--red); font-family: var(--font-hud); font-size: 0.65rem; letter-spacing: 0.12em; cursor: pointer; width: calc(100% - 20px); transition: all 0.2s; }
#sleepBtn:hover { background: rgba(255,58,58,0.1); box-shadow: 0 0 14px rgba(255,58,58,0.25); }

@media (max-width: 900px) {
  body { grid-template-columns: 1fr; grid-template-rows: auto 1fr 35dvh; grid-template-areas: "topbar" "chat" "sidebar"; }
  #topbar { padding: 8px 12px; gap: 12px; overflow-x: auto; flex-wrap: wrap; justify-content: space-between; }
  #status-ticker { display: none; }
  #chat-area { border-right: none; border-bottom: 2px solid var(--border); }
  #activity-bar { overflow-x: auto; white-space: nowrap; }
  #controls-strip { flex-wrap: wrap; gap: 10px; justify-content: space-between; }
  #userInput, select, input[type="number"] { font-size: 16px; }
  .msg { max-width: 92%; }
  #statsGrid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 500px) {
  #statsGrid { grid-template-columns: 1fr; }
  .stat-card.wide { grid-column: 1; }
  .brand { font-size: 0.7rem; }
  .hud-pill { font-size: 0.55rem; padding: 2px 6px; }
  #controls-strip { flex-direction: column; align-items: stretch; }
  .ctrl-group { justify-content: space-between; width: 100%; }
  input[type="range"] { flex: 1; margin: 0 10px; }
}

/* MOE FULL-PAGE DASHBOARD STYLING */
.main-tab {
  font-family: var(--font-hud);
  font-size: 0.72rem;
  font-weight: bold;
  color: var(--text-dim);
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0 16px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
  letter-spacing: 0.08em;
}
.main-tab:hover {
  color: var(--text);
  background: rgba(0,255,136,0.03);
}
.main-tab.active {
  color: var(--g0);
  border-bottom-color: var(--g0);
  background: rgba(0,255,136,0.04);
  text-shadow: 0 0 8px rgba(0,255,136,0.4);
}

#moe-dashboard {
  grid-area: chat;
  display: none;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg2);
  border-right: 2px solid var(--border);
}
.moe-dash-header {
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
}
.moe-dash-title {
  font-family: var(--font-hud);
  font-size: 0.85rem;
  font-weight: bold;
  color: var(--g0);
  letter-spacing: 0.1em;
}
.moe-dash-body {
  display: grid;
  grid-template-columns: 1.25fr 0.75fr;
  flex: 1;
  overflow: hidden;
}
@media (max-width: 1000px) {
  .moe-dash-body {
    grid-template-columns: 1fr;
    grid-template-rows: 1.2fr 0.8fr;
  }
}
.moe-dash-left {
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 16px;
  gap: 16px;
}
.moe-dash-right {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg3);
  padding: 16px;
  gap: 12px;
}
.moe-pipeline-widget {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.moe-grid-registry {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.architect-chat-standalone {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.moe-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
  overflow-y: auto;
  padding-right: 4px;
}
.moe-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  transition: all 0.2s ease;
}
.moe-card:hover {
  border-color: var(--g0);
  box-shadow: 0 0 10px rgba(0,255,136,0.1);
}
.moe-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.moe-card-title {
  font-family: var(--font-mono);
  font-weight: bold;
  color: var(--g0);
  font-size: 0.85rem;
}
.moe-badge {
  font-size: 0.6rem;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: var(--font-mono);
  text-transform: uppercase;
}
.moe-badge.frozen {
  background: rgba(255,153,0,0.15);
  border: 1px solid rgba(255,153,0,0.5);
  color: #ff9900;
}
.moe-badge.mutable {
  background: rgba(0,204,255,0.15);
  border: 1px solid rgba(0,204,255,0.5);
  color: #00ccff;
}
.moe-badge.mcp-on {
  background: rgba(0,255,136,0.12);
  border: 1px solid rgba(0,255,136,0.45);
  color: var(--g0);
}
.moe-badge.mcp-off {
  background: rgba(255,58,58,0.12);
  border: 1px solid rgba(255,58,58,0.45);
  color: var(--red);
}
.prompt-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: border-color 0.2s;
}
.prompt-card:hover { border-color: var(--g3); }
.prompt-card-name {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  font-weight: bold;
  color: var(--g0);
  letter-spacing: 0.04em;
}
.prompt-card-desc {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-dim);
}
.prompt-card-body {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--g2);
  border-top: 1px solid var(--border);
  padding-top: 6px;
  margin-top: 2px;
  white-space: pre-wrap;
  max-height: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.moe-card-desc {
  font-size: 0.75rem;
  color: var(--g2);
  line-height: 1.3;
  margin-bottom: 10px;
}
.moe-card-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  font-size: 0.65rem;
  font-family: var(--font-mono);
  background: var(--bg3);
  padding: 6px;
  border-radius: 4px;
}
.moe-metric-val {
  color: var(--g0);
}
.moe-card-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
}
.moe-pipeline-strip {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 4px;
}
.moe-pipeline-header {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--g2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  letter-spacing: 0.05em;
}
.moe-pipeline-flow {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px 0;
}
.moe-flow-badge {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--g1);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  padding: 4px 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.moe-flow-badge.active-exec {
  border-color: var(--g0);
  background: rgba(0,255,136,0.05);
}
.moe-flow-arrow {
  color: var(--g3);
  font-size: 0.8rem;
}
.moe-routing-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  color: var(--g0);
}
/* Switch styling */
.switch {
  position: relative;
  display: inline-block;
  width: 28px;
  height: 16px;
}
.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}
.slider {
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: var(--bg3);
  border: 1px solid var(--border);
  transition: .2s;
  border-radius: 16px;
}
.slider:before {
  position: absolute;
  content: "";
  height: 10px; width: 10px;
  left: 2px; bottom: 2px;
  background-color: var(--g2);
  transition: .2s;
  border-radius: 50%;
}
input:checked + .slider {
  background-color: rgba(0,255,136,0.1);
  border-color: var(--g0);
}
input:checked + .slider:before {
  transform: translateX(12px);
  background-color: var(--g0);
}

/* Module Builder Chat CSS */
.moe-builder-chat {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  height: 200px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  margin-top: auto;
}
.moe-builder-header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 8px 12px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: bold;
  color: var(--g1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.moe-builder-messages {
  flex: 1;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.moe-builder-msg {
  max-width: 85%;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.72rem;
  line-height: 1.35;
}
.moe-builder-msg.user {
  background: rgba(0,204,255,0.1);
  border: 1px solid rgba(0,204,255,0.3);
  color: #cceeff;
  align-self: flex-end;
}
.moe-builder-msg.bot {
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--g2);
  align-self: flex-start;
}
.moe-deploy-card {
  margin-top: 6px;
  background: rgba(0,255,136,0.05);
  border: 1px dashed var(--g0);
  border-radius: 6px;
  padding: 8px;
  font-size: 0.68rem;
}
.moe-builder-input-area {
  display: flex;
  border-top: 1px solid var(--border);
  background: var(--bg2);
}
.moe-builder-input {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--g0);
  font-family: var(--font-sans);
  font-size: 0.75rem;
  padding: 8px 12px;
  outline: none;
}
.moe-builder-btn {
  background: transparent;
  border: none;
  color: var(--g0);
  cursor: pointer;
  padding: 8px 12px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  transition: color 0.2s;
}
.moe-builder-btn:hover {
  color: #fff;
}
.btn-delete-module {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 3px 6px;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  outline: none;
}
.btn-delete-module:hover {
  border-color: var(--red);
  color: var(--red);
  box-shadow: 0 0 6px rgba(255,58,58,0.25);
}
.modal-overlay {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(8px);
  z-index: 10000;
  justify-content: center;
  align-items: center;
}
.btn-create-expert-from-msg {
  transition: all 0.2s ease;
}
</style>
</head>
<body>

<!-- Beautiful Glassmorphic Expert Creator Modal -->
<div id="expertCreatorModal" class="modal-overlay">
  <div class="modal-content" style="background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; width: 90%; max-width: 500px; padding: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.5); font-family: var(--font-hud);">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 15px;">
      <span style="font-size: 0.95rem; font-weight: bold; color: var(--g0); letter-spacing: 0.05em;">📦 CREATE EXPERT FROM RESPONSE</span>
      <span onclick="closeExpertCreatorModal()" style="color: var(--text-dim); cursor: pointer; font-size: 1.2rem; font-weight: bold;" onmouseover="this.style.color='var(--red)'" onmouseout="this.style.color='var(--text-dim)'">&times;</span>
    </div>
    
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">EXPERT NAME:</label>
        <input type="text" id="modalExpertName" value="Custom Advisor" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; outline: none; font-family: var(--font-mono);" oninput="autoGenModalId(this.value)">
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">EXPERT ID (SLUG):</label>
        <input type="text" id="modalExpertId" value="custom-advisor" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">DESCRIPTION:</label>
        <textarea id="modalExpertDesc" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; height: 50px; resize: none; outline: none; font-family: var(--font-mono);">An expert memory module generated dynamically from a curated response.</textarea>
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">SYSTEM DIRECTIVE / EXPERTISE RULES:</label>
        <textarea id="modalExpertDirective" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; height: 110px; resize: vertical; outline: none; font-family: var(--font-mono);"></textarea>
      </div>
      
      <div style="display: flex; gap: 10px;">
        <div style="flex: 1;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">LTP DECAY:</label>
          <input type="number" id="modalExpertLtp" value="0.95" step="0.01" min="0.5" max="1" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px; font-size: 0.75rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
        </div>
        <div style="flex: 1;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">STP DECAY:</label>
          <input type="number" id="modalExpertStp" value="0.80" step="0.01" min="0.5" max="1" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px; font-size: 0.75rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; padding-left: 10px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">FROZEN:</label>
          <input type="checkbox" id="modalExpertFrozen" style="width: 18px; height: 18px; cursor: pointer;">
        </div>
      </div>
    </div>
    
    <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px; border-top: 1px solid var(--border); padding-top: 15px;">
      <button onclick="closeExpertCreatorModal()" style="background: transparent; border: 1px solid var(--border); color: var(--text-dim); padding: 6px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; font-family: var(--font-mono);" onmouseover="this.style.color='#fff'" onmouseout="this.style.color='var(--text-dim)'">CANCEL</button>
      <button onclick="submitModalCreateExpert()" style="background: var(--g0); border: none; color: #000; padding: 6px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: var(--font-mono);" onmouseover="this.style.boxShadow='0 0 10px var(--g0)'" onmouseout="this.style.boxShadow='none'">⚡ CREATE & DEPLOY</button>
    </div>
  </div>
</div>

<!-- Beautiful Memory Viewer Modal -->
<div id="memoryViewerModal" class="modal-overlay">
  <div class="modal-content" style="background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; width: 90%; max-width: 700px; padding: 25px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); font-family: var(--font-hud); display: flex; flex-direction: column; gap: 15px; max-height: 85vh; overflow-y: auto;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 5px;">
      <span style="font-size: 0.95rem; font-weight: bold; color: var(--g0); letter-spacing: 0.05em; display: flex; align-items: center; gap: 6px;">👁️ VIEW MEMORY DETAILS</span>
      <span onclick="closeMemoryViewerModal()" style="color: var(--text-dim); cursor: pointer; font-size: 1.2rem; font-weight: bold;" onmouseover="this.style.color='var(--red)'" onmouseout="this.style.color='var(--text-dim)'">&times;</span>
    </div>
    
    <div style="display: flex; flex-direction: column; gap: 15px;">
      <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">MEMORY ID:</label>
          <div id="modalMemId" style="background: var(--bg3); border: 1px solid var(--border); padding: 8px; font-size: 0.72rem; border-radius: 4px; font-family: var(--font-mono); word-break: break-all; color: var(--text-dim);"></div>
        </div>
        <div style="flex: 1; min-width: 150px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">MODULE ID:</label>
          <div id="modalMemModule" style="background: var(--bg3); border: 1px solid var(--border); padding: 8px; font-size: 0.75rem; border-radius: 4px; font-family: var(--font-mono); color: var(--g0); font-weight: bold;"></div>
        </div>
      </div>

      <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 150px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">ENERGY TENSORS:</label>
          <div id="modalMemEnergy" style="background: var(--bg3); border: 1px solid var(--border); padding: 8px; font-size: 0.75rem; border-radius: 4px; font-family: var(--font-mono); color: var(--text);"></div>
        </div>
        <div style="flex: 1; min-width: 150px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">TIMESTAMP:</label>
          <div id="modalMemDate" style="background: var(--bg3); border: 1px solid var(--border); padding: 8px; font-size: 0.75rem; border-radius: 4px; font-family: var(--font-mono); color: var(--text-dim);"></div>
        </div>
      </div>

      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">TAGS & CATEGORIES:</label>
        <div id="modalMemTags" style="background: var(--bg3); border: 1px solid var(--border); padding: 8px; font-size: 0.75rem; border-radius: 4px; font-family: var(--font-mono); color: var(--g3); font-weight: bold;"></div>
      </div>

      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">FULL MEMORY TEXT (UNTRUNCATED):</label>
        <div id="modalMemText" style="background: var(--bg3); border: 1px solid var(--border); padding: 12px; font-size: 0.82rem; border-radius: 4px; font-family: var(--font-mono); white-space: pre-wrap; word-break: break-word; color: #fff; max-height: 250px; overflow-y: auto; line-height: 1.5; border-left: 3px solid var(--g0);"></div>
      </div>

      <div id="modalMemImageContainer" style="display: none;">
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">ASSOCIATED IMAGE ARCHIVE:</label>
        <div style="border: 1px solid var(--border); border-radius: 4px; overflow: hidden; display: inline-block; background: var(--bg3); cursor: pointer;" onclick="window.open(document.getElementById('modalMemImg').src, '_blank')">
          <img id="modalMemImg" src="" style="max-width: 100%; max-height: 200px; display: block; filter: brightness(0.92) contrast(1.05);">
        </div>
      </div>
    </div>
    
    <div style="display: flex; justify-content: flex-end; margin-top: 10px; border-top: 1px solid var(--border); padding-top: 15px;">
      <button onclick="closeMemoryViewerModal()" style="background: var(--g0); border: none; color: #000; padding: 6px 20px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: var(--font-mono); letter-spacing: 0.05em;" onmouseover="this.style.boxShadow='0 0 10px var(--g0)'" onmouseout="this.style.boxShadow='none'">DISMISS</button>
    </div>
  </div>
</div>

<div id="topbar">
  <div class="brand">ERN <span>//</span> DGX</div>
  <div class="main-tabs" style="display:flex; height:100%; margin-left:24px; border-left: 1px solid var(--border); padding-left: 8px;">
    <div id="main-tab-chat" class="main-tab active" onclick="switchMainView('chat')">💬 CHAT CONSOLE</div>
    <div id="main-tab-moe" class="main-tab" onclick="switchMainView('moe')">🧠 MOE ARCHITECT</div>
    <div id="main-tab-prompts" class="main-tab" onclick="switchMainView('prompts')">⚡ PROMPTS</div>
  </div>
  <div id="status-ticker"><span id="ticker-inner">SYSTEM ONLINE — PYTORCH ENGINE ACTIVE — AWAITING QUERIES — EPIGENETIC RESONANCE NETWORK INITIALIZED —&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SYSTEM ONLINE — PYTORCH ENGINE ACTIVE — AWAITING QUERIES — EPIGENETIC RESONANCE NETWORK INITIALIZED —&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></div>
  <div class="hud-pill live">VRAM LIVE</div>
  <div class="hud-pill">NODES: <span id="node-count">—</span></div>
  <div class="hud-pill">QUERIES: <span id="query-count-hud">0</span></div>
</div>

<div id="chat-area">
  <div id="activity-bar">
    <div class="phase" id="phase-retrieve"><div class="phase-dot" id="dot-retrieve"></div><span>TENSOR RETRIEVE</span></div>
    <div class="phase" id="phase-gen"><div class="phase-dot" id="dot-gen"></div><span>LLM GENERATE</span></div>
    <div class="phase" id="phase-judge"><div class="phase-dot" id="dot-judge"></div><span>MEMORY JUDGE</span></div>
    <div class="phase" id="phase-encode"><div class="phase-dot" id="dot-encode"></div><span>ENCODE</span></div>
    <span id="activity-text" style="flex:1; text-align:right;"></span>
  </div>

  <div id="chatBox">
    <div class="msg bot">EPIGENETIC RESONANCE NETWORK ONLINE
PyTorch tensor engine bound to CUDA
Awaiting input...</div>
  </div>

  <div id="input-area">
    <button id="uploadBtn" onclick="document.getElementById('pdfInput').click()" style="background:transparent; border:1px solid var(--border); color:var(--text-dim); padding:10px 12px; font-size:1.1rem; cursor:pointer; transition: all 0.15s; outline:none; font-family:var(--font-mono);" title="Upload PDF/Image for Epigenetic Extraction">📎</button>
    <input type="file" id="pdfInput" accept=".pdf, image/*" style="display:none" onchange="handleAttachmentUpload(this)">
    <input type="text" id="userInput" placeholder="// INPUT QUERY..." autocomplete="off" spellcheck="false">
    <button id="sendBtn" onclick="send()">TRANSMIT</button>
  </div>

  <div id="controls-strip">
    <div class="ctrl-group">
      <span class="ctrl-label">FOCUS θ</span>
      <input type="range" id="focusSlider" min="0.05" max="0.45" step="0.05" value="0.15" oninput="document.getElementById('fv').textContent=parseFloat(this.value).toFixed(2)">
      <span class="val-badge" id="fv">0.15</span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">MODEL</span>
      <select id="modelSelect"><option value="qwen2.5-coder:7b">qwen2.5-coder:7b</option></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">AGENTIC SEARCH</span>
      <button id="agenticToggleBtn" onclick="toggleAgenticSearch()" style="background:transparent; border:1px solid var(--g0); color:var(--g0); padding:3px 8px; font-family:var(--font-hud); font-size:0.6rem; cursor:pointer; letter-spacing:0.08em; transition: all 0.15s; outline:none; text-shadow: 0 0 6px var(--g0);">ON</button>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">TEMPORAL DISCOVERY</span>
      <button id="temporalToggleBtn" onclick="toggleTemporalDiscovery()" style="background:transparent; border:1px solid var(--g0); color:var(--g0); padding:3px 8px; font-family:var(--font-hud); font-size:0.6rem; cursor:pointer; letter-spacing:0.08em; transition: all 0.15s; outline:none; text-shadow: 0 0 6px var(--g0);">ON</button>
    </div>
    <div class="ctrl-group" style="margin-left:auto;">
      <span class="ctrl-label" style="color:#3a5a4a;">HIST</span>
      <span class="val-badge" id="hist-len" style="color:var(--text-dim)">0</span>
      <span class="ctrl-label" style="margin-left:8px;">
        <button onclick="clearHistory()" style="background:none;border:none;color:var(--text-dim);font-family:var(--font-hud);font-size:0.58rem;cursor:pointer;letter-spacing:0.08em;">CLR</button>
      </span>
    </div>
  </div>
</div>

<!-- GORGEOUS FULL-PAGE MOE CONTROLLER -->
<div id="moe-dashboard">
  <div class="moe-dash-header">
    <div class="moe-dash-title">🤖 EPIGENETIC MIXTURE-OF-EXPERTS (MOE) CONTROL CENTER</div>
    <div style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-dim);">VRAM ACCELERATED DIVISION</div>
  </div>
  <div class="moe-dash-body">
    <!-- Left column: Registry list and static pipeline settings -->
    <div class="moe-dash-left">
      <!-- Pipeline Strip -->
      <div class="moe-pipeline-widget">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div style="font-family:var(--font-mono); font-size:0.8rem; font-weight:bold; color:var(--g0);">ACTIVE PIPELINE CONFIGURATION</div>
          <div class="moe-routing-pill">
            <span style="font-family:var(--font-mono); font-size:0.75rem; letter-spacing:0.05em; color:var(--g1); margin-right:4px;">AUTO-ROUTE:</span>
            <label class="switch">
              <input type="checkbox" id="moeAutoRoute" onchange="toggleAutoRoute(this.checked)">
              <span class="slider"></span>
            </label>
          </div>
        </div>
        <div id="moePipelineFlow" class="moe-pipeline-flow">
          <div class="no-mem">// INACTIVE PIPELINE</div>
        </div>
      </div>
      
      <!-- Registry Grid Header -->
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
        <div style="font-family:var(--font-hud); font-size:0.85rem; font-weight:bold; color:var(--text);">ACTIVE EXPERT REGISTRY</div>
        <button onclick="loadModulesUI()" style="background:transparent; border:1px solid var(--border); color:var(--g0); font-family:var(--font-mono); font-size:0.7rem; padding:4px 8px; border-radius:4px; cursor:pointer;">🔄 REFRESH REGISTRY</button>
      </div>
      
      <!-- Grid Container for Cards -->
      <div class="moe-grid-registry" id="moeModulesList">
        <div class="no-mem">Loading Modules...</div>
      </div>
    </div>

    <!-- Right column: Massive interactive Builder chat console -->
    <div class="moe-dash-right">
      <div style="font-family:var(--font-hud); font-size:0.8rem; font-weight:bold; color:var(--g0); margin-bottom:10px; letter-spacing:0.05em;">💬 CONSTRUCT NEW EXPERTS WITH AI</div>
      
      <div class="architect-chat-standalone">
        <div class="moe-builder-header">
          <span>🤖 AI MODULE ARCHITECT</span>
          <button onclick="clearBuilderChat()" style="background:none; border:none; color:var(--text-dim); font-size:0.6rem; cursor:pointer; font-family:var(--font-mono);">RESET</button>
        </div>
        <div class="moe-builder-messages" id="builderMessages">
          <div class="moe-builder-msg bot">Greetings. I am the ERN Expert Architect. Converse with me to design and customize a new expert memory module, or configure parameter thresholds. Let me know what you want to build.</div>
        </div>
        <div class="moe-builder-input-area">
          <input type="text" id="builderInput" class="moe-builder-input" placeholder="Say 'make a python coding style module'..." onkeydown="if(event.key==='Enter') sendBuilderMessage()">
          <button onclick="sendBuilderMessage()" class="moe-builder-btn">SEND</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- PROMPTS MANAGER PANEL -->
<div id="prompts-dashboard" style="display:none; flex-direction:column; flex:1; padding:18px 24px; gap:18px; overflow-y:auto;">
  <div class="moe-dash-header">
    <div class="moe-dash-title">⚡ MCP PROMPT REGISTRY</div>
    <div style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-dim);">ZED SLASH COMMAND MANAGER</div>
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; flex:1;">
    <!-- Left: Prompt list -->
    <div style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-family:var(--font-hud); font-size:0.85rem; font-weight:bold; color:var(--text);">DEPLOYED PROMPTS</div>
        <button onclick="loadPromptsUI()" style="background:transparent; border:1px solid var(--border); color:var(--g0); font-family:var(--font-mono); font-size:0.7rem; padding:4px 8px; border-radius:4px; cursor:pointer;">🔄 REFRESH</button>
      </div>
      <div id="promptsList" style="display:flex; flex-direction:column; gap:8px;">
        <div class="no-mem">Loading prompts...</div>
      </div>
    </div>
    <!-- Right: Create / Edit form -->
    <div style="display:flex; flex-direction:column; gap:10px;">
      <div style="font-family:var(--font-hud); font-size:0.85rem; font-weight:bold; color:var(--g0); letter-spacing:0.05em;">✏️ CREATE / EDIT PROMPT</div>
      <div style="background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:14px; display:flex; flex-direction:column; gap:10px;">
        <input type="hidden" id="promptEditingName" value="">
        <div>
          <label style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-dim); display:block; margin-bottom:4px; letter-spacing:0.06em;">NAME (becomes /slash-command)</label>
          <input type="text" id="promptName" placeholder="my_coding_rules" oninput="updateSlugPreview()" style="width:100%; box-sizing:border-box; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-family:var(--font-mono); font-size:0.8rem; padding:8px 10px; border-radius:4px; outline:none;">
          <div id="promptSlugPreview" style="font-family:var(--font-mono); font-size:0.6rem; color:var(--g2); margin-top:4px;">Zed command: <span style="color:var(--g0);">/autonomous_memory_agent</span></div>
        </div>
        <div>
          <label style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-dim); display:block; margin-bottom:4px; letter-spacing:0.06em;">DESCRIPTION (shown as tooltip in Zed)</label>
          <input type="text" id="promptDescription" placeholder="Injects coding rules and style preferences..." style="width:100%; box-sizing:border-box; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-family:var(--font-mono); font-size:0.8rem; padding:8px 10px; border-radius:4px; outline:none;">
        </div>
        <div>
          <label style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-dim); display:block; margin-bottom:4px; letter-spacing:0.06em;">BODY (injected as system message)</label>
          <textarea id="promptBody" rows="10" placeholder="You are a coding assistant with strict rules...\nAlways use async/await..." style="width:100%; box-sizing:border-box; background:var(--bg3); border:1px solid var(--border); color:var(--text); font-family:var(--font-mono); font-size:0.75rem; padding:8px 10px; border-radius:4px; outline:none; resize:vertical; line-height:1.5;"></textarea>
        </div>
        <div style="display:flex; gap:8px;">
          <button id="promptSaveBtn" onclick="savePrompt()" style="flex:1; background:rgba(0,255,136,0.1); border:1px solid var(--g2); color:var(--g0); font-family:var(--font-hud); font-size:0.75rem; padding:10px; border-radius:4px; cursor:pointer; letter-spacing:0.08em; transition: all 0.2s;">⚡ DEPLOY PROMPT</button>
          <button onclick="clearPromptForm()" style="background:transparent; border:1px solid var(--border); color:var(--text-dim); font-family:var(--font-hud); font-size:0.75rem; padding:10px 14px; border-radius:4px; cursor:pointer;">CLR</button>
        </div>
      </div>
    </div>
  </div>
</div>


<div id="sidebar">
  <div style="padding: 10px 10px 0 10px;">
    <div style="font-family:var(--font-mono); font-size:0.6rem; color:var(--g2); margin-bottom:4px; letter-spacing:0.05em;">ACTIVE SIDEPANEL EXPERT:</div>
    <select id="moduleSelectGlobal" onchange="onGlobalModuleChange()" style="width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--g0); font-family:var(--font-mono); font-size:0.75rem; padding:6px; border-radius:4px; outline:none; cursor:pointer;">
      <option value="auto-route" selected>🧠 Dynamic Router (Auto-Route)</option>
    </select>
  </div>

  <div class="panel-tabs" style="margin-top: 8px;">
    <div class="tab active" onclick="switchTab('recall')">RECALL</div>
    <div class="tab" onclick="switchTab('vault')">VAULT</div>
    <div class="tab" onclick="switchTab('deltas')">DELTAS</div>
    <div class="tab" onclick="switchTab('stats')">STATS</div>
  </div>

  <div class="panel-body">
    <div id="tab-recall" class="panel-section visible">
      <canvas id="neuro-canvas"></canvas>
      <div id="memScroll"><div class="no-mem">No active synapses — send a message to retrieve.</div></div>
    </div>

    <div id="tab-vault" class="panel-section">
      <input type="text" id="vaultSearch" placeholder="// FILTER SYNAPSES..." oninput="filterVault()" autocomplete="off" spellcheck="false">
      <div id="vaultScroll"><div class="no-mem">Loading Synaptic Vault...</div></div>
    </div>

    <div id="tab-deltas" class="panel-section">
      <div id="deltaStatsBar">
        <span class="stat-pill"><span class="sp-val sp-ENCODE" id="ds-encode">0</span> ENCODE</span>
        <span class="stat-pill"><span class="sp-val sp-DECAY"  id="ds-decay">0</span>  DECAY</span>
        <span class="stat-pill"><span class="sp-val sp-BOOST"  id="ds-boost">0</span>  BOOST</span>
        <span class="stat-pill"><span class="sp-val sp-SLEEP"  id="ds-sleep">0</span>  SLEEP</span>
        <span class="stat-pill" style="margin-left:auto; color:var(--text-dim);">DEPTH: <span id="ds-depth">—</span></span>
      </div>
      <div id="deltaScroll"></div>
      <div id="rollback-row">
        <span class="ctrl-label">ROLLBACK</span>
        <input type="number" id="rollbackN" value="1" min="1" max="50">
        <span class="ctrl-label">ENCODE(S)</span>
        <button id="rollbackBtn" onclick="doRollback()">↩ UNDO</button>
      </div>
    </div>

    <div id="tab-stats" class="panel-section">
      <div id="statsGrid">
        <div class="stat-card"><div class="sc-label">NETWORK NODES</div><div class="sc-val" id="sc-nodes">—</div></div>
        <div class="stat-card"><div class="sc-label">QUERY COUNT</div><div class="sc-val" id="sc-queries">0</div></div>
        <div class="stat-card"><div class="sc-label">DELTA DEPTH</div><div class="sc-val" id="sc-depth">—</div></div>
        <div class="stat-card"><div class="sc-label">AVG RESONANCE</div><div class="sc-val" id="sc-resonance">—</div></div>
        <div class="stat-card wide">
          <div class="sc-label">MEMORY ENERGY DISTRIBUTION</div>
          <div class="energy-bar-wrap">
            <div class="energy-bar-label"><span>DECAY</span><span id="eb-pct">—</span></div>
            <div class="energy-bar"><div class="energy-bar-fill" id="energy-fill" style="width:0%"></div></div>
          </div>
        </div>
        <div class="stat-card wide"><div class="sc-label">LAST ACTIVITY</div><div id="sc-last-act" style="font-size:0.75rem; color:var(--text-dim); margin-top:4px; line-height:1.6;">—</div></div>
      </div>
      <button id="sleepBtn" onclick="triggerSleep()">⬛ INITIATE REM SLEEP CYCLE</button>
    </div>
  </div>
</div>

<script>
console.log("ERN UI Booting...");

// ── State ──────────────────────────────────────────────────────────────
let chatHistory    = [];
let queryCount     = 0;
let lastResonances = [];
let lastRetrievedMemories = [];
let activeTab      = 'recall';
let activeMainView = 'chat';
let useAgenticSearch = true;
let useTemporalDiscovery = true;
let vaultMemories   = [];
const pendingModules = {};
const DEFAULT_MODEL = "qwen2.5-coder:7b";

function toggleAgenticSearch() {
  useAgenticSearch = !useAgenticSearch;
  const btn = document.getElementById('agenticToggleBtn');
  if (useAgenticSearch) {
    btn.textContent = 'ON';
    btn.style.borderColor = 'var(--g0)';
    btn.style.color = 'var(--g0)';
    btn.style.textShadow = '0 0 6px var(--g0)';
  } else {
    btn.textContent = 'OFF';
    btn.style.borderColor = 'var(--dim)';
    btn.style.color = 'var(--text-dim)';
    btn.style.textShadow = 'none';
  }
  pushToast(`Agentic Search toggled ${useAgenticSearch ? 'ON' : 'OFF'}`);
}

function toggleTemporalDiscovery() {
  useTemporalDiscovery = !useTemporalDiscovery;
  const btn = document.getElementById('temporalToggleBtn');
  if (useTemporalDiscovery) {
    btn.textContent = 'ON';
    btn.style.borderColor = 'var(--g0)';
    btn.style.color = 'var(--g0)';
    btn.style.textShadow = '0 0 6px var(--g0)';
  } else {
    btn.textContent = 'OFF';
    btn.style.borderColor = 'var(--dim)';
    btn.style.color = 'var(--text-dim)';
    btn.style.textShadow = 'none';
  }
  pushToast(`Temporal Discovery toggled ${useTemporalDiscovery ? 'ON' : 'OFF'}`);
}

function toggleCognitionPanel(el) {
  el.parentElement.classList.toggle('open');
}

async function loadVault() {
  try {
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res = await fetch(`/api/memories?module_id=${modId}`);
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    vaultMemories = data.memories || [];
    renderVaultList(vaultMemories);
  } catch (e) {
    document.getElementById('vaultScroll').innerHTML = `<div class="no-mem" style="color:var(--red);">Vault Offline: ${e.message}</div>`;
  }
}

function renderVaultList(mems) {
  const scroll = document.getElementById('vaultScroll');
  scroll.innerHTML = '';
  if (!mems.length) {
    scroll.innerHTML = '<div class="no-mem">No synapses in vault.</div>';
    return;
  }
  const isGlobal = document.getElementById('moduleSelectGlobal').value === 'auto-route';
  mems.forEach(m => {
    const card = document.createElement('div');
    card.className = 'memory-card';
    card.id = `vc-${m.memory_id}`;
    
    let imgHtml = '';
    if (m.image_url) {
      imgHtml = `<div class="mc-image" style="margin-top: 6px; border: 1px solid var(--border); overflow: hidden; max-height: 90px; max-width: 160px; cursor: pointer; border-radius: 2px;" onclick="window.open('${m.image_url}', '_blank')" title="Click to view full image">` +
                `<img src="${m.image_url}" style="width: 100%; height: 100%; object-fit: cover; filter: brightness(0.92) contrast(1.05);" />` +
                `</div>`;
    }
    
    const formattedDate = m.timestamp ? new Date(m.timestamp * 1000).toLocaleString() : 'Date Unknown';
    const moduleBadge = isGlobal && m.module_name
      ? `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,255,136,0.1); border: 1px solid var(--g3); color: var(--g1); padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono);">${m.module_name}</span>`
      : '';
    
    const mType = m.memory_type || 'fact';
    let typeBadgeColor = '#00bcff';
    if (mType === 'question') typeBadgeColor = '#ffb300';
    if (mType === 'instruction') typeBadgeColor = '#00ff88';
    const typeBadge = `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,0,0,0.3); border: 1px solid ${typeBadgeColor}; color: ${typeBadgeColor}; padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono); text-transform: uppercase;">${mType}</span>`;
    const previewText = m.text.length > 500 ? m.text.slice(0, 500) + '...' : m.text;
    card.innerHTML = `<div class="mc-tags">${m.tags}${moduleBadge}${typeBadge}</div>` +
                     `<div style="word-break: break-word; line-height: 1.4;">${previewText}</div>` +
                     imgHtml +
                     `<div class="mc-energy">${m.energy.toFixed(3)} LTP | ${m.stp_energy ? m.stp_energy.toFixed(3) : '0.000'} STP</div>` +
                     `<div class="mc-date" style="font-size: 0.58rem; color: var(--text-dim); margin-top: 4px; font-family: var(--font-hud);">${formattedDate}</div>` +
                     `<div style="display:flex; gap:6px; margin-top: 6px; width: 100%;">` +
                     `<button style="background: rgba(0,255,136,0.05); border: 1px solid var(--g3); color: var(--g0); font-family: var(--font-hud); font-size: 0.55rem; padding: 6px; cursor: pointer; flex: 1; letter-spacing: 0.08em; transition: all 0.15s; outline: none; text-transform: uppercase;" onclick="viewMemoryDetails('${m.memory_id}')">VIEW DETAILS</button>` +
                     `<button style="background: rgba(255,0,0,0.05); border: 1px solid var(--red); color: var(--text-dim); font-family: var(--font-hud); font-size: 0.55rem; padding: 6px; cursor: pointer; flex: 1; letter-spacing: 0.08em; transition: all 0.15s; outline: none; text-transform: uppercase;" onmouseover="this.style.color='var(--red)'; this.style.textShadow='0 0 6px var(--red)';" onmouseout="this.style.color='var(--text-dim)'; this.style.textShadow='none';" onclick="forgetMemory('${m.memory_id}', '${m.module_id || ''}')">FORGET</button>` +
                     `</div>`;
    scroll.appendChild(card);
  });
}

function filterVault() {
  const q = document.getElementById('vaultSearch').value.toLowerCase().trim();
  if (!q) {
    renderVaultList(vaultMemories);
    return;
  }
  const filtered = vaultMemories.filter(m => 
    m.text.toLowerCase().includes(q) || m.tags.toLowerCase().includes(q)
  );
  renderVaultList(filtered);
}

async function forgetMemory(memory_id, moduleId = null) {
  if (!confirm('Are you sure you want to forget/revert this memory synapse permanently?')) return;
  const modId = moduleId || document.getElementById('moduleSelectGlobal').value;
  const effectiveModId = (modId === 'auto-route') ? 'default-memory' : modId;
  try {
    const res = await fetch(`/api/memory/${memory_id}?module_id=${effectiveModId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete rejected');
    const data = await res.json();
    if (data.error) throw new Error(data.status);
    
    pushToast('Synapse forgotten successfully.');
    
    const card = document.getElementById(`vc-${memory_id}`);
    if (card) {
      card.style.transition = 'all 0.3s ease';
      card.style.opacity = '0';
      card.style.transform = 'translateX(20px)';
      setTimeout(() => card.remove(), 300);
    }
    
    const recallCard = document.getElementById(`rc-${memory_id}`);
    if (recallCard) {
      recallCard.style.transition = 'all 0.3s ease';
      recallCard.style.opacity = '0';
      recallCard.style.transform = 'translateX(20px)';
      setTimeout(() => recallCard.remove(), 300);
    }
    
    vaultMemories = vaultMemories.filter(m => m.memory_id !== memory_id);
    refreshStats();
  } catch (e) {
    pushToast(`Forget failed: ${e.message}`);
  }
}

function viewMemoryDetails(memoryId) {
  let mem = vaultMemories.find(m => m.memory_id === memoryId);
  if (!mem) {
    mem = lastRetrievedMemories.find(m => m.memory_id === memoryId);
  }
  if (!mem) {
    pushToast("Memory details not found in active session cache.", true);
    return;
  }
  
  document.getElementById('modalMemId').textContent = mem.memory_id;
  document.getElementById('modalMemModule').textContent = mem.module_id || document.getElementById('moduleSelectGlobal').value;
  document.getElementById('modalMemTags').textContent = mem.tags || 'No tags associated';
  document.getElementById('modalMemText').textContent = mem.text;
  
  const ltp = mem.energy !== undefined ? mem.energy.toFixed(3) : '0.000';
  const stp = mem.stp_energy !== undefined ? mem.stp_energy.toFixed(3) : '0.000';
  const resonance = mem.resonance !== undefined ? ` | ${mem.resonance.toFixed(3)} Resonance (R)` : '';
  document.getElementById('modalMemEnergy').textContent = `${ltp} LTP | ${stp} STP${resonance}`;
  
  const formattedDate = mem.timestamp ? new Date(mem.timestamp * 1000).toLocaleString() : 'Date Unknown';
  document.getElementById('modalMemDate').textContent = formattedDate;
  
  const imgCont = document.getElementById('modalMemImageContainer');
  const imgElem = document.getElementById('modalMemImg');
  if (mem.image_url) {
    imgElem.src = mem.image_url;
    imgCont.style.display = 'block';
  } else {
    imgElem.src = '';
    imgCont.style.display = 'none';
  }
  
  document.getElementById('memoryViewerModal').style.display = 'flex';
}

function closeMemoryViewerModal() {
  document.getElementById('memoryViewerModal').style.display = 'none';
}

// ── Neural canvas ────────────────────────────────
const canvas  = document.getElementById('neuro-canvas');
const ctx     = canvas.getContext('2d');
let sparkData = new Array(80).fill(0);

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function drawSparkline() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const w = canvas.width, h = canvas.height;
  if(w === 0 || h === 0) { requestAnimationFrame(drawSparkline); return; }

  ctx.strokeStyle = 'rgba(0,255,136,0.05)';
  ctx.lineWidth = 1;
  for(let y=0; y<h; y+=h/4) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

  const grad = ctx.createLinearGradient(0,0,0,h);
  grad.addColorStop(0,  'rgba(0,255,136,0.25)');
  grad.addColorStop(1,  'rgba(0,255,136,0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  const step = w / (sparkData.length - 1);
  ctx.moveTo(0, h);
  sparkData.forEach((v,i) => {
    const x = i * step;
    const y = h - (v / 1.5) * h * 0.85 - 2;
    i === 0 ? ctx.lineTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur  = 6;
  ctx.beginPath();
  sparkData.forEach((v,i) => {
    const x = i * step;
    const y = h - (v / 1.5) * h * 0.85 - 2;
    i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  requestAnimationFrame(drawSparkline);
}
drawSparkline();

function setTickerSpeed(ms) {
  const ticker = document.getElementById('ticker-inner');
  if(ticker) ticker.style.animationDuration = ms + 'ms';
}
setTickerSpeed(22000);

let activityLog = [];
function setPhase(phase, state, label) {
  const dot = document.getElementById('dot-' + phase);
  if(!dot) return;
  dot.className = 'phase-dot' + (state !== 'idle' ? ' ' + state : '');
  const txt = document.getElementById('activity-text');
  if(state === 'active' && label) {
    txt.textContent = '▶ ' + label;
    activityLog.push(label);
    document.getElementById('sc-last-act').textContent = activityLog.slice(-4).reverse().join('\n') || '—';
    setTickerSpeed(8000);
  }
  if(state === 'idle') { txt.textContent = ''; setTickerSpeed(22000); }
}

function resetPhases() {
  ['retrieve','gen','judge','encode'].forEach(p => setPhase(p,'idle'));
}

async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    const sel  = document.getElementById('modelSelect');
    if (!data.models?.length) return;
    sel.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === DEFAULT_MODEL) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) { console.warn('Model list skipped (using default).'); }
}

async function loadDeltas() {
  try {
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res  = await fetch(`/api/deltas?n=40&module_id=${modId}`);
    if (!res.ok) return;
    const data = await res.json();
    const s    = data.stats;
    document.getElementById('ds-encode').textContent = s.by_op.ENCODE;
    document.getElementById('ds-decay' ).textContent = s.by_op.DECAY;
    document.getElementById('ds-boost' ).textContent = s.by_op.BOOST;
    document.getElementById('ds-sleep' ).textContent = s.by_op.SLEEP;
    document.getElementById('ds-depth' ).textContent = s.depth;
    document.getElementById('sc-depth' ).textContent = s.depth;

    const list = document.getElementById('deltaScroll');
    list.innerHTML = '';
    [...data.deltas].reverse().forEach(d => {
      const div   = document.createElement('div');
      div.className = `delta-entry ${d.op}`;
      const parts = d.summary.split('|');
      const detail = parts.slice(1).join('|').trim();
      div.innerHTML = `<span class="de-op">${d.op}</span><span class="de-body">${detail}</span><span class="de-size">${d.prev_size}→${d.next_size}</span>`;
      list.appendChild(div);
    });
  } catch(e) { console.warn('Delta fetch error:', e); }
}

async function doRollback() {
  const n    = parseInt(document.getElementById('rollbackN').value) || 1;
  const modId = document.getElementById('moduleSelectGlobal').value;
  try {
    const res  = await fetch(`/api/deltas/rollback?n=${n}&module_id=${modId}`, { method:'POST' });
    if (!res.ok) throw new Error('Rollback rejected');
    const data = await res.json();
    pushToast(`${data.status} — engine: ${data.engine_size} nodes`);
    document.getElementById('node-count').textContent = data.engine_size;
    loadDeltas();
  } catch(e) {
    pushToast('Rollback failed to execute.');
  }
}

async function triggerSleep() {
  const modId = document.getElementById('moduleSelectGlobal').value;
  document.getElementById('sleepBtn').textContent = '⬛ REM IN PROGRESS...';
  try {
    const res  = await fetch(`/api/system/sleep?module_id=${modId}`, { method:'POST' });
    if (!res.ok) throw new Error('Failed');
    const data = await res.json();
    pushToast(data.status);
    refreshStats();
  } catch(e) {
    pushToast('Sleep cycle failed.');
  } finally {
    document.getElementById('sleepBtn').textContent = '⬛ INITIATE REM SLEEP CYCLE';
  }
}

async function refreshStats() {
  try {
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res  = await fetch(`/api/deltas?n=1&module_id=${modId}`);
    if (!res.ok) return;
    const data = await res.json();
    const latest = data.deltas[0];
    if (latest) {
      document.getElementById('sc-nodes').textContent   = latest.next_size;
      document.getElementById('node-count').textContent = latest.next_size;
    }
    document.getElementById('sc-queries').textContent  = queryCount;
    document.getElementById('query-count-hud').textContent = queryCount;
  } catch(e) {}
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.textContent.toLowerCase() === tab || t.getAttribute('onclick').includes(tab));
  });
  document.querySelectorAll('.panel-section').forEach(s => s.classList.remove('visible'));
  document.getElementById('tab-' + tab).classList.add('visible');
  if (tab === 'vault')  loadVault();
  if (tab === 'deltas') loadDeltas();
  if (tab === 'stats')  refreshStats();
}

function switchMainView(view) {
  activeMainView = view;
  const chatArea      = document.getElementById('chat-area');
  const moeDash       = document.getElementById('moe-dashboard');
  const promptsDash   = document.getElementById('prompts-dashboard');
  const chatTab       = document.getElementById('main-tab-chat');
  const moeTab        = document.getElementById('main-tab-moe');
  const promptsTab    = document.getElementById('main-tab-prompts');

  // Hide all panels
  chatArea.style.display    = 'none';
  moeDash.style.display     = 'none';
  promptsDash.style.display = 'none';
  chatTab.classList.remove('active');
  moeTab.classList.remove('active');
  promptsTab.classList.remove('active');

  if (view === 'moe') {
    moeDash.style.display = 'flex';
    moeTab.classList.add('active');
    loadModulesUI();
  } else if (view === 'prompts') {
    promptsDash.style.display = 'flex';
    promptsTab.classList.add('active');
    loadPromptsUI();
  } else {
    chatArea.style.display = 'flex';
    chatTab.classList.add('active');
  }
}

// ── Prompt Manager ───────────────────────────────────────────────────────────

function _slugifyPromptName(name) {
  return name.toLowerCase().trim().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
}

function updateSlugPreview() {
  const name = document.getElementById('promptName').value;
  const slug = _slugifyPromptName(name) || 'my_prompt';
  const editing = document.getElementById('promptEditingName').value;
  const label = editing ? 'Editing' : 'Zed command';
  document.getElementById('promptSlugPreview').innerHTML =
    `${label}: <span style="color:var(--g0);">/${slug}</span>`;
}

async function loadPromptsUI() {
  try {
    const res = await fetch('/api/prompts');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    renderPromptsList(data.prompts || []);
  } catch(e) {
    document.getElementById('promptsList').innerHTML =
      `<div class="no-mem" style="color:var(--red);">Offline: ${e.message}</div>`;
  }
}

function renderPromptsList(prompts) {
  const container = document.getElementById('promptsList');
  container.innerHTML = '';

  // Always show the hardcoded autonomous_memory_agent as a read-only locked card
  const lockedCard = document.createElement('div');
  lockedCard.className = 'prompt-card';
  lockedCard.style.borderColor = 'rgba(0,255,136,0.3)';
  lockedCard.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div class="prompt-card-name">⚡ /autonomous_memory_agent</div>
      <span class="moe-badge frozen" style="font-size:0.55rem;">CORE</span>
    </div>
    <div class="prompt-card-desc">Activate ERN Autonomous Mode — the master memory directive.</div>
    <div class="prompt-card-body" style="color:var(--g3);">[Hardcoded — edit in mcp_server.py]</div>
  `;
  container.appendChild(lockedCard);

  if (!prompts.length) {
    const empty = document.createElement('div');
    empty.className = 'no-mem';
    empty.textContent = 'No custom prompts deployed yet. Create one →';
    container.appendChild(empty);
    return;
  }

  prompts.forEach(p => {
    const card = document.createElement('div');
    card.className = 'prompt-card';
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div class="prompt-card-name">/ ${p.name}</div>
        <div style="display:flex; gap:6px;">
          <button onclick="editPrompt(${JSON.stringify(p)})"
            style="background:transparent; border:1px solid var(--g3); color:var(--g1); font-family:var(--font-mono); font-size:0.6rem; padding:2px 7px; border-radius:3px; cursor:pointer;">EDIT</button>
          <button onclick="deletePrompt('${p.name}')"
            style="background:transparent; border:1px solid var(--red); color:var(--text-dim); font-family:var(--font-mono); font-size:0.6rem; padding:2px 7px; border-radius:3px; cursor:pointer;"
            onmouseover="this.style.color='var(--red)'" onmouseout="this.style.color='var(--text-dim)'">DEL</button>
        </div>
      </div>
      <div class="prompt-card-desc">${p.description || '(no description)'}</div>
      <div class="prompt-card-body">${p.text ? p.text.slice(0, 200) + (p.text.length > 200 ? '...' : '') : ''}</div>
    `;
    container.appendChild(card);
  });
}

function editPrompt(p) {
  document.getElementById('promptEditingName').value = p.name;
  document.getElementById('promptName').value = p.name;
  document.getElementById('promptName').disabled = true;  // Name is the key — can't change it
  document.getElementById('promptDescription').value = p.description || '';
  document.getElementById('promptBody').value = p.text || '';
  document.getElementById('promptSaveBtn').textContent = '✏️ SAVE CHANGES';
  updateSlugPreview();
}

function clearPromptForm() {
  document.getElementById('promptEditingName').value = '';
  document.getElementById('promptName').value = '';
  document.getElementById('promptName').disabled = false;
  document.getElementById('promptDescription').value = '';
  document.getElementById('promptBody').value = '';
  document.getElementById('promptSaveBtn').textContent = '⚡ DEPLOY PROMPT';
  document.getElementById('promptSlugPreview').innerHTML =
    'Zed command: <span style="color:var(--g0);">/my_prompt</span>';
}

async function savePrompt() {
  const editingName = document.getElementById('promptEditingName').value;
  const name        = _slugifyPromptName(document.getElementById('promptName').value);
  const description = document.getElementById('promptDescription').value.trim();
  const text        = document.getElementById('promptBody').value.trim();

  if (!name) { pushToast('Name is required!', true); return; }
  if (!text)  { pushToast('Body cannot be empty!', true); return; }

  try {
    let res;
    if (editingName) {
      // PATCH existing
      res = await fetch(`/api/prompts/${editingName}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description, text })
      });
    } else {
      // POST new
      res = await fetch('/api/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, text })
      });
    }
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    pushToast(editingName ? `Prompt '/${editingName}' updated!` : `Prompt '/${name}' deployed to Zed!`);
    clearPromptForm();
    loadPromptsUI();
  } catch(e) {
    pushToast(`Failed: ${e.message}`, true);
  }
}

async function deletePrompt(name) {
  if (!confirm(`Delete custom prompt '/${name}'? It will immediately disappear from Zed's / menu.`)) return;
  try {
    const res = await fetch(`/api/prompts/${name}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete rejected');
    pushToast(`Prompt '/${name}' removed from Zed!`);
    loadPromptsUI();
  } catch(e) {
    pushToast(`Error: ${e.message}`, true);
  }
}


function pushToast(msg, isError = false) {
  const t = document.createElement('div');
  t.textContent = msg;
  const borderColor = isError ? 'var(--red)' : 'var(--g2)';
  const color = isError ? 'var(--red)' : 'var(--g0)';
  const shadow = isError ? 'rgba(255,77,77,0.2)' : 'rgba(0,255,136,0.2)';
  Object.assign(t.style, {
    position:'fixed', bottom:'60px', right:'16px', zIndex:'9998',
    background:'var(--bg3)', border:`1px solid ${borderColor}`,
    color: color, fontFamily:'var(--font-mono)', fontSize:'0.75rem',
    padding:'8px 14px', animation:'msg-in 0.2s ease',
    boxShadow:`0 0 14px ${shadow}`,
    maxWidth:'300px',
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

async function handleAttachmentUpload(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  
  const isPDF = file.name.toLowerCase().endsWith('.pdf');
  const isImage = /\.(png|jpe?g|webp|bmp)$/i.test(file.name) || file.type.startsWith('image/');
  
  if (!isPDF && !isImage) {
    pushToast('Unsupported file format. Please upload a PDF or an Image.', true);
    return;
  }

  const uploadBtn = document.getElementById('uploadBtn');
  const origText = uploadBtn.textContent;
  
  uploadBtn.disabled = true;
  uploadBtn.textContent = '⏳';
  uploadBtn.style.color = 'var(--amber)';
  uploadBtn.style.borderColor = 'var(--amber)';

  const modId = document.getElementById('moduleSelectGlobal').value;
  const endpoint = (isPDF ? '/api/memory/upload-pdf' : '/api/memory/upload-image') + `?module_id=${modId}`;
  const label = isPDF ? 'PDF' : 'Image';
  pushToast(`Ingesting ${label}: ${file.name}... Sending to VRAM targeting ${modId}.`, false);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Network error or file too large.');
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    pushToast(`${label} transmitted successfully! Subconscious fact extraction commenced in background.`, false);
    setTimeout(loadVault, 5000);
  } catch (e) {
    pushToast(`${label} Extraction Trigger Failed: ${e.message}`, true);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = origText;
    uploadBtn.style.color = 'var(--text-dim)';
    uploadBtn.style.borderColor = 'var(--border)';
    input.value = '';
  }
}

function clearHistory() {
  chatHistory = [];
  document.getElementById('hist-len').textContent = '0';
  pushToast('Conversation history cleared.');
}

async function send() {
  const text = document.getElementById('userInput').value.trim();
  if (!text) return;

  const chatBox = document.getElementById('chatBox');
  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = text;
  chatBox.appendChild(uDiv);
  document.getElementById('userInput').value = '';
  document.getElementById('sendBtn').disabled = true;

  const loadDiv = document.createElement('div');
  loadDiv.className = 'msg bot thinking';
  loadDiv.id = 'load';
  loadDiv.innerHTML = 'RETRIEVING SYNAPSES<span class="thinking-dots"></span>';
  chatBox.appendChild(loadDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  setPhase('retrieve', 'active', 'Tensor cosine retrieval...');

  try {
    setTimeout(() => {
      setPhase('retrieve','done');
      setPhase('gen','active','Generating response via LLM...');
      const ld = document.getElementById('load');
      if(ld) ld.innerHTML = 'GENERATING<span class="thinking-dots"></span>';
    }, 300);

    const selectedExpert = document.getElementById('moduleSelectGlobal') ? document.getElementById('moduleSelectGlobal').value : 'auto-route';
    const isAutoRoute = (selectedExpert === 'auto-route');
    const customPipeline = isAutoRoute ? getActivePipeline() : [selectedExpert];

    const res  = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        message            : text,
        model              : document.getElementById('modelSelect').value,
        history            : chatHistory.slice(-6),
        focus_threshold    : parseFloat(document.getElementById('focusSlider').value),
        agentic_search     : useAgenticSearch,
        temporal_discovery : useTemporalDiscovery,
        auto_route         : isAutoRoute,
        pipeline           : customPipeline,
        active_expert      : isAutoRoute ? null : selectedExpert
      })
    });
    
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    setPhase('gen','done');
    setPhase('judge','active','Memory judge analysing facts...');
    setTimeout(() => {
      setPhase('judge','done');
      setPhase('encode','active','Encoding new synapses...');
      setTimeout(() => { setPhase('encode','done'); resetPhases(); }, 1500);
    }, 1200);

    const ld = document.getElementById('load');
    if(ld) ld.remove();

    const bDiv = document.createElement('div');
    bDiv.className = 'msg bot';
    
    const replyText = document.createElement('div');
    
    let formattedReply = data.reply
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    
    formattedReply = formattedReply.replace(/!\[(.*?)\]\((.*?)\)/g, 
      '<div class="chat-inline-image" style="margin: 8px 0; border: 1px solid var(--border); overflow: hidden; max-width: 320px; border-radius: 4px; cursor: pointer;" onclick="window.open(\'$2\', \'_blank\')" title="Click to view full image">' +
      '<img src="$2" alt="$1" style="width:100%; height:auto; display:block; filter: brightness(0.92) contrast(1.05);" />' +
      '</div>'
    );
    
    formattedReply = formattedReply.replace(/\[(.*?)\]\((.*?)\)/g, 
      '<a href="$2" target="_blank" style="color: var(--g0); text-decoration: underline; font-family: var(--font-mono); font-size: 0.72rem;">$1</a>'
    );
    
    formattedReply = formattedReply.replace(/\n/g, '<br>');
    
    replyText.innerHTML = formattedReply;
    bDiv.appendChild(replyText);

    const actionDiv = document.createElement('div');
    actionDiv.className = 'msg-actions';
    actionDiv.style.marginTop = '8px';
    actionDiv.style.display = 'flex';
    actionDiv.style.justifyContent = 'flex-end';
    actionDiv.innerHTML = `
      <button class="btn-create-expert-from-msg" style="background: rgba(0,255,127,0.06); border: 1px solid rgba(0,255,127,0.4); color: var(--g0); font-family: var(--font-mono); font-size: 0.65rem; padding: 4px 8px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.2s; font-weight: bold; letter-spacing: 0.03em;" onmouseover="this.style.background='var(--g0)'; this.style.color='#000'; this.style.borderColor='var(--g0)';" onmouseout="this.style.background='rgba(0,255,127,0.06)'; this.style.color='var(--g0)'; this.style.borderColor='rgba(0,255,127,0.4)';" onclick="openCreateExpertModalFromMsg(this)">
        ➕ CREATE EXPERT FROM RESPONSE
      </button>
    `;
    bDiv.appendChild(actionDiv);

    const steps = data.agentic_steps || [];
    if (steps.length > 0) {
      const monitor = document.createElement('div');
      monitor.className = 'cognition-monitor';
      
      const header = document.createElement('div');
      header.className = 'cognition-header';
      header.innerHTML = `[▶ COGNITIVE CYCLE STEPS: ${steps.length} PHASE(S)]`;
      header.setAttribute('onclick', 'toggleCognitionPanel(this)');
      monitor.appendChild(header);
      
      const body = document.createElement('div');
      body.className = 'cognition-body';
      
      steps.forEach(s => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'cognition-step';
        stepDiv.innerHTML = `<span class="step-dot ${s.status}"></span>` +
                            `<div><span class="step-title">${s.step}:</span>` +
                            `<span class="step-detail">${s.detail}</span></div>`;
        body.appendChild(stepDiv);
      });
      
      monitor.appendChild(body);
      bDiv.appendChild(monitor);
    }

    chatBox.appendChild(bDiv);

    chatHistory.push({role:'user',content:text},{role:'assistant',content:data.reply});
    queryCount++;
    document.getElementById('hist-len').textContent = Math.floor(chatHistory.length / 2);
    document.getElementById('sc-queries').textContent   = queryCount;
    document.getElementById('query-count-hud').textContent = queryCount;
    chatBox.scrollTop = chatBox.scrollHeight;

    const memories = data.memories || [];
    const resValues = memories.map(m => m.resonance);
    const avgRes = resValues.length ? resValues.reduce((a,b)=>a+b,0)/resValues.length : 0;
    
    sparkData.push(avgRes);
    sparkData = sparkData.slice(-80);
    lastResonances = resValues;
    lastRetrievedMemories = memories;

    document.getElementById('sc-resonance').textContent = avgRes > 0 ? avgRes.toFixed(3) : '—';

    const pct = Math.min(100, (avgRes / 1.5) * 100);
    document.getElementById('energy-fill').style.width = pct + '%';
    document.getElementById('eb-pct').textContent = pct.toFixed(0) + '%';

    const mBox = document.getElementById('memScroll');
    if (memories.length > 0) {
      mBox.innerHTML = '';
      memories.forEach((m) => {
        const card = document.createElement('div');
        card.className = 'memory-card';
        card.id = `rc-${m.memory_id}`;
        
        let imgHtml = '';
        if (m.image_url) {
          imgHtml = `<div class="mc-image" style="margin-top: 6px; border: 1px solid var(--border); overflow: hidden; max-height: 90px; max-width: 160px; cursor: pointer; border-radius: 2px;" onclick="window.open('${m.image_url}', '_blank')" title="Click to view full image">` +
                    `<img src="${m.image_url}" style="width: 100%; height: 100%; object-fit: cover; filter: brightness(0.92) contrast(1.05);" />` +
                    `</div>`;
        }
        
        const formattedDate = m.timestamp ? new Date(m.timestamp * 1000).toLocaleString() : 'Date Unknown';
        
        const mType = m.memory_type || 'fact';
        let typeBadgeColor = '#00bcff';
        if (mType === 'question') typeBadgeColor = '#ffb300';
        if (mType === 'instruction') typeBadgeColor = '#00ff88';
        const typeBadge = `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,0,0,0.3); border: 1px solid ${typeBadgeColor}; color: ${typeBadgeColor}; padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono); text-transform: uppercase;">${mType}</span>`;
        
        const previewText = m.text.length > 500 ? m.text.slice(0, 500) + '...' : m.text;
        card.innerHTML = `<div class="mc-tags">[Expert: ${m.module_name || 'default-memory'}] | ${m.tags}${typeBadge}</div>` +
                         `<div style="word-break: break-word; line-height: 1.4;">${previewText}</div>` +
                         imgHtml +
                         `<div class="mc-resonance">${m.resonance.toFixed(3)} R</div>` +
                         `<div class="mc-energy">${m.energy ? m.energy.toFixed(3) : '0.000'} LTP | ${m.stp_energy ? m.stp_energy.toFixed(3) : '0.000'} STP</div>` +
                         `<div class="mc-date" style="font-size: 0.58rem; color: var(--text-dim); margin-top: 4px; font-family: var(--font-hud);">${formattedDate}</div>` +
                         `<div style="display:flex; gap:6px; margin-top: 6px; width: 100%;">` +
                         `<button style="background: rgba(0,255,136,0.05); border: 1px solid var(--g3); color: var(--g0); font-family: var(--font-hud); font-size: 0.55rem; padding: 6px; cursor: pointer; flex: 1; letter-spacing: 0.08em; transition: all 0.15s; outline: none; text-transform: uppercase;" onclick="viewMemoryDetails('${m.memory_id}')">VIEW DETAILS</button>` +
                         `<button style="background: rgba(255,0,0,0.05); border: 1px solid var(--red); color: var(--text-dim); font-family: var(--font-hud); font-size: 0.55rem; padding: 6px; cursor: pointer; flex: 1; letter-spacing: 0.08em; transition: all 0.15s; outline: none; text-transform: uppercase;" onmouseover="this.style.color='var(--red)'; this.style.textShadow='0 0 6px var(--red)';" onmouseout="this.style.color='var(--text-dim)'; this.style.textShadow='none';" onclick="forgetMemory('${m.memory_id}', '${m.module_id || 'default-memory'}')">FORGET</button>` +
                         `</div>`;
        mBox.appendChild(card);
      });
    } else {
      mBox.innerHTML = '<div class="no-mem">No synapses resonated above threshold.</div>';
    }

    if (activeTab === 'deltas') loadDeltas();
    if (activeTab === 'vault')  loadVault();
    if (activeMainView === 'moe') loadModulesUI();
    refreshStats();

  } catch(e) {
    const ld = document.getElementById('load');
    if(ld) ld.remove();
    setPhase('retrieve','error','Error'); setPhase('gen','error');
    setTimeout(resetPhases, 2000);
    
    const eDiv = document.createElement('div');
    eDiv.className = 'msg bot';
    eDiv.style.borderLeftColor = 'var(--red)';
    eDiv.textContent = 'SYSTEM ERROR: API unreachable (' + e.message + ')';
    chatBox.appendChild(eDiv);
  } finally {
    document.getElementById('sendBtn').disabled = false;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

document.getElementById('userInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// ── Mixture of Experts (MOE) Logic ───────────────────────────────────────────
let moeModules = [];
let defaultPipeline = [];
let builderChatHistory = [];

async function onGlobalModuleChange() {
  const modId = document.getElementById('moduleSelectGlobal').value;
  pushToast(`Active sidepanel expert switched to: ${modId}`);
  
  const autoRouteChk = document.getElementById('moeAutoRoute');
  if (autoRouteChk) {
    if (modId === 'auto-route') {
      autoRouteChk.checked = true;
    } else {
      autoRouteChk.checked = false;
    }
    renderPipelineFlow();
  }

  if (activeTab === 'vault')  loadVault();
  if (activeTab === 'deltas') loadDeltas();
  if (activeTab === 'stats')  refreshStats();
}

async function loadModulesUI() {
  try {
    const res = await fetch('/api/modules');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    moeModules = data.modules || [];
    defaultPipeline = data.default_pipeline || [];
    
    const globalSel = document.getElementById('moduleSelectGlobal');
    const currentVal = globalSel.value || 'auto-route';
    globalSel.innerHTML = '';
    
    const routeOpt = document.createElement('option');
    routeOpt.value = 'auto-route';
    routeOpt.textContent = '🧠 Dynamic Router (Auto-Route)';
    routeOpt.selected = (currentVal === 'auto-route');
    globalSel.appendChild(routeOpt);

    moeModules.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.config.module_id;
      opt.textContent = `${m.config.name} (${m.config.module_id})`;
      if (m.config.module_id === currentVal) opt.selected = true;
      globalSel.appendChild(opt);
    });
    
    renderPipelineFlow();
    renderModulesList();

  } catch(e) {
    document.getElementById('moeModulesList').innerHTML = `<div class="no-mem" style="color:var(--red);">Offline: ${e.message}</div>`;
  }
}

function renderPipelineFlow() {
  const flow = document.getElementById('moePipelineFlow');
  flow.innerHTML = '';
  
  if (document.getElementById('moeAutoRoute').checked) {
    flow.innerHTML = `<div class="moe-flow-badge active-exec" style="border-color:var(--g0); color:var(--g0);">🧠 Dynamic Router Deciding...</div>`;
    return;
  }
  
  if (!defaultPipeline.length) {
    flow.innerHTML = `<div class="no-mem">// PIPELINE EMPTY</div>`;
    return;
  }
  
  defaultPipeline.forEach((pid, idx) => {
    const m = moeModules.find(x => x.config.module_id === pid);
    if (!m) return;
    
    const badge = document.createElement('div');
    badge.className = 'moe-flow-badge';
    badge.id = `flow-badge-${pid}`;
    badge.textContent = m.config.name;
    
    flow.appendChild(badge);
    
    if (idx < defaultPipeline.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'moe-flow-arrow';
      arrow.innerHTML = '⚡';
      flow.appendChild(arrow);
    }
  });
}

function getActivePipeline() {
  if (document.getElementById('moeAutoRoute').checked) {
    return [];
  }
  return defaultPipeline;
}

async function toggleAutoRoute(checked) {
  pushToast(`Auto-Route ${checked ? 'enabled (Dynamic Routing)' : 'disabled (Static Pipeline)'}`);
  const globalSel = document.getElementById('moduleSelectGlobal');
  if (globalSel) {
    if (checked) {
      globalSel.value = 'auto-route';
    } else {
      globalSel.value = 'default-memory';
    }
  }
  renderPipelineFlow();
}

function renderModulesList() {
  const container = document.getElementById('moeModulesList');
  container.innerHTML = '';
  
  moeModules.forEach(m => {
    const c = m.config;
    const card = document.createElement('div');
    card.className = 'moe-card';
    
    const isDefault = c.module_id === 'default-memory';
    const isPipeline = defaultPipeline.includes(c.module_id);
    const badgeClass = c.frozen ? 'frozen' : 'mutable';
    const badgeText = c.frozen ? 'FROZEN' : 'MUTABLE';
    
    const deleteBtn = isDefault ? '' : `<button onclick="deleteModule('${c.module_id}')" class="btn-delete-module">🗑️ DELETE</button>`;
    const pipelineCheckbox = `
      <input type="checkbox" id="chk-${c.module_id}" ${isPipeline ? 'checked' : ''} onchange="togglePipelineModule('${c.module_id}', this.checked)" style="cursor:pointer;">
    `;
    
    card.innerHTML = `
      <div class="moe-card-header">
        <div style="display:flex; align-items:center; gap:8px;">
          ${pipelineCheckbox}
          <span class="moe-card-title">${c.name}</span>
        </div>
        <div style="display:flex; gap:6px; align-items:center;">
          <span class="moe-badge ${badgeClass}" onclick="toggleFrozenState('${c.module_id}', ${!c.frozen})" style="cursor:pointer; user-select:none; transition: all 0.2s;" title="Click to toggle frozen status (frozen experts prevent synaptic decay)">${badgeText}</span>
          <span class="moe-badge ${c.mcp_enabled !== false ? 'mcp-on' : 'mcp-off'}" onclick="toggleMcpState('${c.module_id}', ${c.mcp_enabled === false})" style="cursor:pointer; user-select:none; transition: all 0.2s;" title="Expose/Hide this expert to external MCP agents (like Zed)">${c.mcp_enabled !== false ? 'MCP' : 'NO MCP'}</span>
        </div>
      </div>
      <div class="moe-card-desc">${c.description}</div>
      <div class="moe-card-metrics">
        <div>SYNAPSES: <span class="moe-metric-val">${m.synapses_count}</span></div>
        <div>LTP DECAY: <span class="moe-metric-val">${c.ltp_decay_rate.toFixed(2)}</span></div>
        <div>STP DECAY: <span class="moe-metric-val">${c.stp_decay_rate.toFixed(2)}</span></div>
      </div>
      <div class="moe-card-actions">
        <span style="font-size:0.6rem; color:var(--text-dim); font-family:var(--font-mono);">${c.module_id}</span>
        ${deleteBtn}
      </div>
    `;
    container.appendChild(card);
  });
}

async function toggleFrozenState(moduleId, isFrozen) {
  try {
    pushToast(`${isFrozen ? 'Freezing' : 'Unfreezing'} expert module '${moduleId}'...`);
    const res = await fetch(`/api/modules/${moduleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frozen: isFrozen })
    });
    if (!res.ok) throw new Error('API modification failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully ${isFrozen ? 'froze' : 'unfroze'} '${moduleId}'!`);
    loadModulesUI();
  } catch(e) {
    pushToast(`Failed to toggle state: ${e.message}`, true);
    loadModulesUI();
  }
}

async function toggleMcpState(moduleId, isMcpEnabled) {
  try {
    pushToast(`${isMcpEnabled ? 'Enabling' : 'Disabling'} MCP for expert module '${moduleId}'...`);
    const res = await fetch(`/api/modules/${moduleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mcp_enabled: isMcpEnabled })
    });
    if (!res.ok) throw new Error('API modification failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully ${isMcpEnabled ? 'exposed' : 'hidden'} '${moduleId}' ${isMcpEnabled ? 'to' : 'from'} MCP!`);
    loadModulesUI();
  } catch(e) {
    pushToast(`Failed to toggle MCP state: ${e.message}`, true);
    loadModulesUI();
  }
}

async function togglePipelineModule(moduleId, checked) {
  let newPipeline = [...defaultPipeline];
  if (checked) {
    if (!newPipeline.includes(moduleId)) newPipeline.push(moduleId);
  } else {
    newPipeline = newPipeline.filter(pid => pid !== moduleId);
  }
  
  try {
    const res = await fetch('/api/modules/pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newPipeline)
    });
    if (!res.ok) throw new Error('Failed to update pipeline');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    defaultPipeline = data.default_pipeline || [];
    pushToast(`Pipeline updated: ${defaultPipeline.join(' -> ')}`);
    renderPipelineFlow();
  } catch(e) {
    pushToast(`Error: ${e.message}`, true);
    loadModulesUI();
  }
}

async function deleteModule(moduleId) {
  if (!confirm(`Are you sure you want to delete module '${moduleId}' permanently? All synapses will be scrubbed.`)) return;
  try {
    const res = await fetch(`/api/modules/${moduleId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('API delete failed');
    pushToast(`Module '${moduleId}' deleted successfully.`);
    loadModulesUI();
  } catch(e) {
    pushToast(`Error: ${e.message}`, true);
  }
}

async function sendBuilderMessage() {
  const input = document.getElementById('builderInput');
  const text = input.value.trim();
  if (!text) return;
  
  const messagesBox = document.getElementById('builderMessages');
  
  const uDiv = document.createElement('div');
  uDiv.className = 'moe-builder-msg user';
  uDiv.textContent = text;
  messagesBox.appendChild(uDiv);
  input.value = '';
  messagesBox.scrollTop = messagesBox.scrollHeight;
  
  const loadDiv = document.createElement('div');
  loadDiv.className = 'moe-builder-msg bot';
  loadDiv.innerHTML = 'Thinking<span class="thinking-dots"></span>';
  messagesBox.appendChild(loadDiv);
  messagesBox.scrollTop = messagesBox.scrollHeight;
  
  try {
    const res = await fetch('/api/modules/builder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: builderChatHistory
      })
    });
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    
    loadDiv.remove();
    
    builderChatHistory.push({ role: 'user', content: text });
    builderChatHistory.push({ role: 'assistant', content: data.reply });
    
    const bDiv = document.createElement('div');
    bDiv.className = 'moe-builder-msg bot';
    
    let reply = data.reply;
    
    const jsonMatch = reply.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    let deployCardHtml = '';
    if (jsonMatch) {
      try {
        let jsonStr = jsonMatch[1].replace(/\[?MODULE_READY\]?/gi, '').trim();
        
        const startIdx = jsonStr.indexOf('{');
        if (startIdx !== -1) {
          let endIdx = Math.max(jsonStr.lastIndexOf('}'), jsonStr.lastIndexOf(']'));
          if (endIdx !== -1 && endIdx > startIdx) {
            jsonStr = jsonStr.substring(startIdx, endIdx + 1);
          }
        }
        
        if (jsonStr.startsWith('{') && jsonStr.endsWith(']')) {
          jsonStr = jsonStr.slice(0, -1) + '}';
        }
        
        jsonStr = jsonStr.replace(/\/\/.*$/gm, '');
        jsonStr = jsonStr.replace(/,\s*([\}\]])/g, '$1');
        
        const config = JSON.parse(jsonStr);
        const configId = 'mod_' + Math.random().toString(36).substring(2, 9);
        pendingModules[configId] = config;
        deployCardHtml = `
          <div class="moe-deploy-card">
            <strong style="color:var(--g0);">📦 EXPERT READY TO DEPLOY</strong><br/>
            Name: ${config.name || 'Unnamed'}<br/>
            ID: ${config.module_id || 'unnamed-expert'}<br/>
            Directives: ${config.system_directive ? config.system_directive.slice(0, 40) + '...' : 'None'}<br/>
            <button onclick="deployBuiltModule('${configId}')" style="background:var(--g0); border:none; color:#000; font-family:var(--font-mono); font-size:0.65rem; padding:4px 8px; border-radius:3px; margin-top:6px; cursor:pointer; font-weight:bold;">⚡ DEPLOY EXPERT</button>
          </div>
        `;
      } catch(je) {
        console.warn("JSON parsing in architect reply failed:", je);
      }
    }
    
    bDiv.innerHTML = `<div style="white-space:pre-wrap;">${reply}</div>` + deployCardHtml;
    messagesBox.appendChild(bDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    
  } catch(e) {
    if (loadDiv) loadDiv.remove();
    const bDiv = document.createElement('div');
    bDiv.className = 'moe-builder-msg bot';
    bDiv.style.borderColor = 'var(--red)';
    bDiv.textContent = `Architect failed: ${e.message}`;
    messagesBox.appendChild(bDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }
}

async function deployBuiltModule(configId) {
  try {
    const config = pendingModules[configId];
    if (!config) throw new Error('Configuration not found');
    pushToast(`Deploying expert module '${config.name}'...`);
    const res = await fetch('/api/modules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) throw new Error('API creation failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully deployed '${config.name}'!`);
    loadModulesUI();
    
    const messagesBox = document.getElementById('builderMessages');
    const sDiv = document.createElement('div');
    sDiv.className = 'moe-builder-msg bot';
    sDiv.style.borderLeftColor = 'var(--g0)';
    sDiv.innerHTML = `<span style="color:var(--g0);">✔️ SUCCESS:</span> Module '${config.name}' deployed, loaded in PyTorch memory, and active.`;
    messagesBox.appendChild(sDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    
  } catch(e) {
    pushToast(`Deployment Failed: ${e.message}`, true);
  }
}

function clearBuilderChat() {
  builderChatHistory = [];
  document.getElementById('builderMessages').innerHTML = '<div class="moe-builder-msg bot">Greetings. I am the ERN Expert Architect. Converse with me to design and customize a new expert memory module, or configure parameter thresholds.</div>';
}

function autoGenModalId(name) {
  const slug = name.toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
  document.getElementById('modalExpertId').value = slug;
}

function openCreateExpertModalFromMsg(btn) {
  const botMsgDiv = btn.closest('.msg.bot');
  const clone = botMsgDiv.cloneNode(true);
  const actionsDiv = clone.querySelector('.msg-actions');
  if (actionsDiv) actionsDiv.remove();
  const monitorDiv = clone.querySelector('.cognition-monitor');
  if (monitorDiv) monitorDiv.remove();
  
  let text = clone.innerText.trim();
  
  document.getElementById('modalExpertDirective').value = `Act according to these guidelines and rules:\n${text}`;
  
  const words = text.split(/\s+/).slice(0, 3).join(' ');
  const cleanName = words.replace(/[^a-zA-Z0-9\s]+/g, '').trim() || 'Custom Expert';
  document.getElementById('modalExpertName').value = cleanName + ' Expert';
  autoGenModalId(cleanName + ' Expert');
  
  document.getElementById('expertCreatorModal').style.display = 'flex';
}

function closeExpertCreatorModal() {
  document.getElementById('expertCreatorModal').style.display = 'none';
}

async function submitModalCreateExpert() {
  const name = document.getElementById('modalExpertName').value.trim();
  const id = document.getElementById('modalExpertId').value.trim();
  const desc = document.getElementById('modalExpertDesc').value.trim();
  const directive = document.getElementById('modalExpertDirective').value.trim();
  const ltp = parseFloat(document.getElementById('modalExpertLtp').value);
  const stp = parseFloat(document.getElementById('modalExpertStp').value);
  const frozen = document.getElementById('modalExpertFrozen').checked;
  
  if (!name || !id) {
    pushToast("Name and ID are required fields!", true);
    return;
  }
  
  try {
    pushToast(`Creating expert module '${id}'...`);
    const res = await fetch('/api/modules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        module_id: id,
        name: name,
        description: desc,
        frozen: frozen,
        ltp_decay_rate: ltp,
        stp_decay_rate: stp,
        sleep_threshold: 0.10,
        focus_threshold: 0.15,
        system_directive: directive
      })
    });
    
    if (!res.ok) throw new Error('API request failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully deployed Expert '${name}'!`);
    closeExpertCreatorModal();
    
    setTimeout(() => {
      loadModulesUI().then(() => {
        const globalSel = document.getElementById('moduleSelectGlobal');
        globalSel.value = id;
        onGlobalModuleChange();
      });
    }, 500);
  } catch(e) {
    pushToast(`Failed to deploy expert: ${e.message}`, true);
  }
}

loadModels();
loadModulesUI();
refreshStats();
setInterval(() => {
  sparkData.push(Math.random() * 0.05);
  sparkData = sparkData.slice(-80);
}, 2000);

let _registryFingerprint = '';
async function _pollRegistryChanges() {
  try {
    const res = await fetch('/api/modules');
    if (!res.ok) return;
    const data = await res.json();
    const fp = JSON.stringify({
      ids: (data.modules || []).map(m => m.config.module_id).sort(),
      pipeline: data.default_pipeline || [],
      synapseCounts: (data.modules || []).map(m => `${m.config.module_id}:${m.synapses_count}`)
    });
    if (fp !== _registryFingerprint) {
      if (_registryFingerprint !== '') {
        moeModules = data.modules || [];
        defaultPipeline = data.default_pipeline || [];
        
        const globalSel = document.getElementById('moduleSelectGlobal');
        const currentVal = globalSel.value;
        globalSel.innerHTML = '';
        const routeOpt = document.createElement('option');
        routeOpt.value = 'auto-route';
        routeOpt.textContent = '🧠 Dynamic Router (Auto-Route)';
        routeOpt.selected = (currentVal === 'auto-route');
        globalSel.appendChild(routeOpt);
        moeModules.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.config.module_id;
          opt.textContent = `${m.config.name} (${m.config.module_id})`;
          if (m.config.module_id === currentVal) opt.selected = true;
          globalSel.appendChild(opt);
        });

        renderPipelineFlow();
        renderModulesList();
        if (activeTab === 'vault')  loadVault();
        if (activeTab === 'stats')  refreshStats();
        pushToast('Registry updated — UI reloaded.');
      }
      _registryFingerprint = fp;
    }
  } catch(_) {}
}
setInterval(_pollRegistryChanges, 3000);
</script>
</body>
</html>
"""
