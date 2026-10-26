"""Build a self-contained HTML form for blind human rubric scoring of judging items.

    python src/make_label_form.py data/judge_items/pilot.jsonl --n 12 --out labeling/pilot_form.html

Open the HTML file in Chrome/Edge (double-click). Candidates are shown in a random order with neutral
letters; which model/prompt wrote each one is hidden. Progress autosaves in the browser.
"Download results" saves one JSON file -> put it in labeling/results/ (the browser will ask where to save).
The same rubric as the AI judges (src/judge_prompt.py) and the automatic checks (src/auto_rubric.py).
"""
import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompt import PERSONAS as JP_PERSONAS, RUBRIC  # noqa: E402  (same rubric text as the AI judges)

DIMS = [("fidelity", "Numbers match the data"), ("units", "Units & scale right"),
        ("direction", "Increase/decrease & calculations right"), ("attribution", "Right period & entity"),
        ("invented", "Nothing invented (causes, comparisons, verdicts)"), ("coverage", "Key facts covered")]
PERSONAS = [("value_investor", "Value investor"), ("fund_manager", "Fund manager"), ("retail_investor", "Retail investor")]

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind rubric scoring</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#ddd;--acc:#2f5d8a;--bad:#a33}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Arial}
header{position:sticky;top:0;background:var(--ink);color:#fff;padding:8px 16px;display:flex;gap:12px;align-items:center;z-index:5;flex-wrap:wrap}
header button{background:#fff;color:var(--ink);border:0;border-radius:6px;padding:6px 10px;cursor:pointer}
main{display:grid;grid-template-columns:minmax(300px,38%) 1fr;gap:12px;padding:12px 16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
main>section:first-child{position:sticky;top:52px;align-self:start}@media(max-width:900px){main>section:first-child{position:static}}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,Consolas,monospace;margin:0;max-height:calc(100vh - 130px);overflow:auto}
.cand{margin-bottom:14px}.cand h3{margin:0 0 6px;color:var(--acc)}
.text{white-space:pre-wrap;background:#fafaf8;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:13px}
table{border-collapse:collapse;margin-top:8px;font-size:12.5px}td,th{padding:3px 6px;border-bottom:1px solid #eee;text-align:left}
td input[type=radio]{margin:0 3px}.sev{color:var(--bad);font-weight:600}
input[type=text]{width:100%;box-sizing:border-box;padding:4px;border:1px solid var(--line);border-radius:4px}
summary{cursor:pointer}#guide pre{font:13px/1.45 system-ui,Segoe UI,Arial;margin-top:8px}
.num.hit{background:#ffe066;outline:1px solid #e0b000;border-radius:2px}
.warn{color:var(--bad);font-weight:600}.mute{color:var(--mute)}
</style></head><body>
<header><b>Blind rubric scoring</b><span id="pos"></span><button id="prev">◀ Prev</button><button id="next">Next ▶</button>
<button id="dl">Download results</button><span id="status" class="mute"></span></header>
<details id="guide" style="margin:10px 16px 0" class="panel"><summary><b>Scoring guide</b> (same rubric the AI judges get; click to open/close)</summary><pre style="max-height:none">__GUIDE__</pre></details>
<main><section class="panel"><h2 style="margin-top:0">Data</h2><div id="hint" class="mute" style="min-height:1.2em;font-size:12px">Hover over a number in a summary to find it here.</div><pre id="card"></pre></section>
<section class="panel" id="cands"></section></main>
<script>
const ITEMS=__ITEMS__, DIMS=__DIMS__, PERS=__PERS__, KEY="blindform:"+__FORMID__;
// saved progress (browser localStorage), current item index
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=st._i||0, t0=Date.now();
const save=()=>{try{st._i=i;localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
// per-item record of the rater's answers
function rec(it){return st[it.form_id]=st[it.form_id]||{seconds:0,scores:{},usefulness:{},overall:{},rank:{},severe:{},errors:{}}}
function radios(name,val,on){let h="";for(let v=1;v<=5;v++)h+=`<label><input type=radio name="${name}" value=${v} ${val==v?"checked":""} data-on="${on}">${v}</label>`;return h}
// number lookup: highlight numbers in the data panel that match the hovered number (rounding allowed)
const esc=t=>t.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const numspan=h=>h.replace(/-?\d[\d,]*(?:\.\d+)?/g,m=>`<span class=num data-n="${m.replace(/,/g,"").replace(/^-/,"")}">${m}</span>`);
function hl(n){document.querySelectorAll(".num.hit").forEach(e=>e.classList.remove("hit"));if(!n)return;
 const v=parseFloat(n);let first=null;document.querySelectorAll("#card .num").forEach(e=>{const w=parseFloat(e.dataset.n);
  if(e.dataset.n===n||(isFinite(v)&&isFinite(w)&&Math.abs(w-v)<=Math.max(0.051,Math.abs(w)*0.0005))){e.classList.add("hit");first=first||e}});
 document.querySelectorAll(`#cands .num[data-n="${n}"]`).forEach(e=>e.classList.add("hit"));
 const box=document.getElementById("card");if(first)box.scrollTop=first.offsetTop-box.offsetTop-60;
 document.getElementById("hint").textContent=first?"":`${n}: not found in the data (may be computed, rounded differently, or wrong)`}
document.addEventListener("mouseover",e=>{if(e.target.matches("#cands .num"))hl(e.target.dataset.n)});
document.addEventListener("click",e=>{if(e.target.matches("#cands .num"))hl(e.target.dataset.n)});
// draw the current item: data card, then each candidate with its rubric inputs
function render(){
 const it=ITEMS[i], r=rec(it); document.getElementById("pos").textContent=`Item ${i+1} / ${ITEMS.length}  (${it.symbol} ${it.snapshot})`;
 document.getElementById("card").innerHTML=numspan(esc(it.card)); let h="";
 it.cands.forEach(c=>{const L=c.letter;
  h+=`<div class=cand><h3>Summary ${L}</h3><div class=text>${c.html}</div><table><tr><th>Dimension (1 worst – 5 best)</th><th></th></tr>`;
  DIMS.forEach(([d,lab])=>{h+=`<tr><td>${lab}</td><td>${radios(it.form_id+L+d,(r.scores[L]||{})[d],`s|${L}|${d}`)}</td></tr>`});
  PERS.forEach(([p,lab])=>{h+=`<tr><td>Useful for: ${lab}</td><td>${radios(it.form_id+L+p,(r.usefulness[p]||{})[L],`u|${L}|${p}`)}</td></tr>`});
  h+=`<tr><td><b>Overall</b></td><td>${radios(it.form_id+L+"ov",r.overall[L],`o|${L}`)}</td></tr>
   <tr><td><b>Rank</b> (1 = best, no ties)</td><td>${[1,2,3,4,5,6].slice(0,it.cands.length).map(v=>`<label><input type=radio name="${it.form_id+L}rk" value=${v} ${r.rank[L]==v?"checked":""} data-on="r|${L}">${v}</label>`).join("")}</td></tr>
   <tr><td class=sev>Severe error?</td><td><label><input type=checkbox data-on="v|${L}" ${r.severe[L]?"checked":""}> wrong revenue / net-profit figure or direction (incl. turnaround called a decline), 1000x scale, sector figure as company's (overall must be 1-2)</label></td></tr>
   <tr><td>Errors found</td><td><input type=text data-on="e|${L}" value="${(r.errors[L]||"").replace(/"/g,"&quot;")}"></td></tr></table></div>`});
 h+=`<p id=check></p>`; document.getElementById("cands").innerHTML=h; check(); t0=Date.now(); window.scrollTo(0,0)}
// completeness check: missing fields, duplicate ranks, severe error with overall > 2
function check(){const it=ITEMS[i],r=rec(it),n=it.cands.length;let miss=[];
 it.cands.forEach(c=>{const L=c.letter;DIMS.forEach(([d])=>{if(!(r.scores[L]||{})[d])miss.push(L+":"+d)});if(!r.overall[L])miss.push(L+":overall");if(!r.rank[L])miss.push(L+":rank")});
 const ranks=Object.values(r.rank); const dup=ranks.length&&new Set(ranks).size!==ranks.length;
 const sevBad=it.cands.filter(c=>r.severe[c.letter]&&r.overall[c.letter]>2).map(c=>c.letter);
 document.getElementById("check").innerHTML=(miss.length?`<span class=mute>Missing: ${miss.length} fields</span> `:"✔ complete ")+(dup?`<span class=warn>Ranks must be different. </span>`:"")+(sevBad.length?`<span class=warn>Severe error but overall > 2 for ${sevBad.join(",")}. </span>`:"");
 document.getElementById("status").textContent=`${ITEMS.filter(x=>{const q=st[x.form_id];return q&&Object.keys(q.rank).length===x.cands.length}).length}/${ITEMS.length} ranked`}
document.addEventListener("change",e=>{const on=e.target.dataset.on;if(!on)return;const it=ITEMS[i],r=rec(it),[k,L,x]=on.split("|"),v=e.target.type==="checkbox"?e.target.checked:(e.target.type==="text"?e.target.value:+e.target.value);
 if(k==="s"){(r.scores[L]=r.scores[L]||{})[x]=v}else if(k==="u"){(r.usefulness[x]=r.usefulness[x]||{})[L]=v}else if(k==="o"){r.overall[L]=v}else if(k==="r"){r.rank[L]=v}else if(k==="v"){r.severe[L]=v}else if(k==="e"){r.errors[L]=v}
 save();check()});
function leave(){const r=rec(ITEMS[i]);r.seconds+=Math.round((Date.now()-t0)/1000);save()}
document.getElementById("prev").onclick=()=>{leave();i=Math.max(0,i-1);render()};
document.getElementById("next").onclick=()=>{leave();i=Math.min(ITEMS.length-1,i+1);render()};
// download all answers as JSONL in the same format as the AI judges (letters mapped back to candidate ids)
document.getElementById("dl").onclick=()=>{leave();const out=ITEMS.map(it=>{const r=st[it.form_id]||{};const m=Object.fromEntries(it.cands.map(c=>[c.letter,c.cand_id]));
  const tr=o=>Object.fromEntries(Object.entries(o||{}).map(([L,v])=>[m[L],v]));
  return {item_id:it.item_id,order:"human",judge:"human",seconds:r.seconds||0,
   ranking:Object.entries(r.rank||{}).sort((a,b)=>a[1]-b[1]).map(([L])=>m[L]),overall:tr(r.overall),
   scores:Object.fromEntries(Object.entries(r.scores||{}).map(([L,v])=>[m[L],{...v,severe_error:!!(r.severe||{})[L],errors:(r.errors||{})[L]?[(r.errors||{})[L]]:[]}])),
   usefulness:Object.fromEntries(Object.entries(r.usefulness||{}).map(([p,v])=>[p,tr(v)])),letter_to_cand:m,saved_at:new Date().toISOString()}});
 const b=new Blob([out.map(o=>JSON.stringify(o)).join("\n")+"\n"],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=__FORMID__+"_results.jsonl";a.click()};
render();
</script></body></html>"""


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_html(text: str) -> str:
    """HTML-escape a summary and wrap every number in a span, so hovering it highlights the same number in the data."""
    return NUM.sub(lambda m: f'<span class=num data-n="{m.group(0).replace(",", "").lstrip("-")}">{m.group(0)}</span>',
                   html.escape(text))


# ---- command line: shuffle items and candidates (fixed seed), write the self-contained HTML form
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--n", type=int, help="first N items after shuffling (default all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(items)
    items = items[:args.n] if args.n else items
    form_id = Path(args.out).stem
    out = []
    for k, it in enumerate(items):
        cands = list(it["candidates"])
        rng.shuffle(cands)
        out.append({"form_id": f"{form_id}_{k}", "item_id": it["item_id"], "symbol": it["symbol"], "snapshot": it["snapshot"],
                    "card": it["card"],
                    "cands": [{"letter": chr(65 + j), "cand_id": c["cand_id"], "html": to_html(c["text"])}
                              for j, c in enumerate(cands)]})
    page = (PAGE.replace("__ITEMS__", json.dumps(out, ensure_ascii=False))
            .replace("__DIMS__", json.dumps(DIMS)).replace("__PERS__", json.dumps(PERSONAS))
            .replace("__FORMID__", json.dumps(form_id))
            .replace("__GUIDE__", html.escape(RUBRIC.format(personas="\n".join(f"  {k}: {v}" for k, v in JP_PERSONAS.items())))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"{len(out)} items -> {args.out}  (open it in Chrome or Edge)")
