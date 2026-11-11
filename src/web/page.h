// The browser chat page (single file, no external resources).
#pragma once
static const char* const kWebPage = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dopplers Bakery</title>
<style>
:root{--bg:#ffffff;--fg:#1f2328;--mut:#6b7280;--panel:#f4f4f5;--bub:#eaf1ff;--line:#e4e4e7;--acc:#2563eb;--acc2:#1d4ed8;--err:#b42318;--shadow:0 2px 12px rgba(0,0,0,.08)}
@media(prefers-color-scheme:dark){:root{--bg:#17181b;--fg:#ececf1;--mut:#9ca3af;--panel:#23252a;--bub:#26324d;--line:#33363d;--acc:#4f8bff;--acc2:#3b76ea;--err:#ff8a80;--shadow:0 2px 12px rgba(0,0,0,.4)}}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:16px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif;display:flex;flex-direction:column}
header{padding:12px 20px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--line);font-weight:600}
header .dot{width:10px;height:10px;border-radius:50%;background:var(--acc)}
header span.sub{color:var(--mut);font-weight:400;font-size:13px}
#log{flex:1;overflow-y:auto;padding:20px 0 8px}
.wrap{max-width:780px;margin:0 auto;padding:0 16px}
.msg{margin:14px 0;display:flex;flex-direction:column}
.msg.user{align-items:flex-end}.msg.assistant,.msg.result{align-items:flex-start}
.bubble{max-width:88%;white-space:pre-wrap;word-wrap:break-word;padding:10px 14px;border-radius:18px}
.user .bubble{background:var(--bub);border-bottom-right-radius:6px}
.assistant .bubble{padding-left:2px}
.note{color:var(--err);font-size:14px;margin:8px 0}
.media{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.media img{max-width:min(420px,100%);max-height:420px;border-radius:14px;display:block;cursor:zoom-in;box-shadow:var(--shadow)}
.user .media{justify-content:flex-end}
.media audio{width:min(380px,100%);height:40px}
.thinking{display:inline-flex;gap:4px;padding:8px 2px}
.thinking i{width:7px;height:7px;border-radius:50%;background:var(--mut);animation:b 1.2s infinite ease-in-out}
.thinking i:nth-child(2){animation-delay:.15s}.thinking i:nth-child(3){animation-delay:.3s}
@keyframes b{0%,60%,100%{opacity:.25;transform:translateY(0)}30%{opacity:1;transform:translateY(-3px)}}
.status{color:var(--mut);font-size:13px;margin:2px 2px 0}
.empty{text-align:center;color:var(--mut);margin-top:18vh}
.empty h2{color:var(--fg);font-weight:600;margin:0 0 6px}
footer{padding:8px 16px 18px}
#composer{max-width:780px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:26px;padding:10px 10px 8px 12px;box-shadow:var(--shadow);transition:border-color .15s}
#composer.drag{border-color:var(--acc);outline:2px dashed var(--acc);outline-offset:3px}
#chips{display:flex;gap:8px;flex-wrap:wrap;padding:2px 2px 8px}
#chips:empty{display:none}
.chip{position:relative;width:64px;height:64px;border-radius:12px;overflow:hidden;border:1px solid var(--line);background:var(--bg)}
.chip img{width:100%;height:100%;object-fit:cover;display:block}
.chip.up{opacity:.5}
.chip button{position:absolute;top:2px;right:2px;width:20px;height:20px;border-radius:50%;border:0;background:rgba(0,0,0,.65);color:#fff;font-size:13px;line-height:20px;padding:0;cursor:pointer}
.row{display:flex;align-items:flex-end;gap:6px}
textarea{flex:1;resize:none;border:0;outline:0;background:transparent;color:var(--fg);font:inherit;max-height:200px;padding:7px 2px;min-height:38px}
.rb{width:38px;height:38px;border-radius:50%;border:0;cursor:pointer;display:grid;place-items:center;flex:none}
#plus{background:transparent;color:var(--fg);font-size:26px;font-weight:300}
#plus:hover{background:var(--line)}
#send{background:var(--acc);color:#fff}#send:hover{background:var(--acc2)}
#send:disabled{background:var(--line);color:var(--mut);cursor:default}
#voice{background:transparent;color:var(--mut)}#voice.on{color:var(--acc)}#voice:hover{background:var(--line)}
#menu{position:fixed;display:none;min-width:230px;background:var(--bg);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:6px;z-index:5}
#menu button,#menu label{display:flex;width:100%;align-items:center;gap:10px;padding:9px 12px;border:0;background:transparent;color:var(--fg);font:inherit;border-radius:9px;cursor:pointer;text-align:left}
#menu button:hover,#menu label:hover{background:var(--panel)}
#menu hr{border:0;border-top:1px solid var(--line);margin:5px 0}
#menu select{margin-left:auto;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:3px 6px;font:inherit}
#big{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;place-items:center;z-index:9;cursor:zoom-out}
#big img{max-width:94vw;max-height:94vh;border-radius:8px}
</style></head><body>
<header><div class="dot"></div>Dopplers Bakery <span class="sub">chat &mdash; images, music, sound effects, speech</span></header>
<div id="log"><div class="wrap" id="msgs"></div></div>
<footer>
 <div id="composer">
  <div id="chips"></div>
  <div class="row">
   <button class="rb" id="plus" title="Add">+</button>
   <textarea id="inp" rows="1" placeholder="Ask for an image, a song or a sound... drop or paste a picture to edit it"></textarea>
   <button class="rb" id="voice" title="Voice replies"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/></svg></button>
   <button class="rb" id="send" title="Send"><svg id="sendIcon" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4l7 8h-4.5v8h-5v-8H5z"/></svg></button>
  </div>
 </div>
</footer>
<div id="menu">
 <button id="mUp"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.5l-8.5 8.5a6 6 0 0 1-8.5-8.5L13 3.5a4 4 0 0 1 5.7 5.7l-9 9a2 2 0 0 1-2.8-2.8l8.3-8.3"/></svg>Upload a picture&hellip;</button>
 <label><input type="checkbox" id="mVoice">&nbsp; Voice replies <select id="mSpk"><option>Ryan</option><option>Aiden</option><option>Vivian</option><option>Serena</option><option>Uncle_Fu</option><option>Dylan</option><option>Eric</option><option>Ono_Anna</option><option>Sohee</option></select></label>
 <hr><button id="mNew"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>New chat</button>
</div>
<input type="file" id="file" accept="image/*" multiple hidden>
<div id="big"><img alt=""></div>
<script>
const $=id=>document.getElementById(id);
const H={'X-Bakery':'1'};
let st=null,attached=[],busy=false,seen=-1,lastSig='';const played=new Set();
const post=(u,b)=>fetch(u,{method:'POST',headers:{...H,'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
function el(t,c,txt){const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e}
function media(files){
 if(!files||!files.length)return null;const m=el('div','media');
 files.forEach(f=>{
  if(f.kind==='image'){const i=el('img');i.src=f.url;i.loading='lazy';i.onclick=()=>{$('big').firstElementChild.src=f.url;$('big').style.display='grid'};m.appendChild(i)}
  else if(f.kind==='audio'){const a=el('audio');a.controls=true;a.preload='metadata';a.src=f.url;a.dataset.auto='1';m.appendChild(a)}
 });return m}
function render(s){
 const sig=JSON.stringify([s.messages,s.live,s.busy,s.status]);if(sig===lastSig)return;lastSig=sig;
 const box=$('msgs'),log=$('log');const near=log.scrollHeight-log.scrollTop-log.clientHeight<140;
 box.innerHTML='';
 if(!s.messages.length&&!s.busy){const e=el('div','empty');e.appendChild(el('h2',null,'What shall we make today?'));e.appendChild(el('div',null,'Describe a picture, a song, a sound effect — or attach a picture and say how to change it.'));box.appendChild(e)}
 s.messages.forEach((m,i)=>{
  if(m.role==='note'){box.appendChild(el('div','note',m.text));return}
  const d=el('div','msg '+m.role);
  if(m.text){d.appendChild(el('div','bubble',m.text))}
  const md=media(m.files);if(md)d.appendChild(md);box.appendChild(d);
  // voice replies play in the browser once, when they first appear
  if(m.role==='assistant'){d.querySelectorAll('audio[data-auto]').forEach(a=>{const u=a.getAttribute('src');if(!played.has(u)){played.add(u);if(seen>=0&&s.voice)a.play().catch(()=>{})}})}
 });
 if(s.busy){const d=el('div','msg assistant');if(s.live)d.appendChild(el('div','bubble',s.live));else{const t=el('div','thinking');t.innerHTML='<i></i><i></i><i></i>';d.appendChild(t)}
  if(s.status)d.appendChild(el('div','status',s.status));box.appendChild(d)}
 seen=s.messages.length;
 if(near||s.busy)log.scrollTop=log.scrollHeight;
}
function applyBusy(b){busy=b;const sb=$('send');sb.disabled=false;
 $('sendIcon').innerHTML=b?'<rect x="6" y="6" width="12" height="12" rx="2"/>':'<path d="M12 4l7 8h-4.5v8h-5v-8H5z"/>';sb.title=b?'Stop':'Send'}
async function poll(){
 try{const r=await fetch('/api/state',{headers:H});st=await r.json();
  if(document.activeElement!==$('mSpk')){$('mVoice').checked=st.voice;$('mSpk').value=st.speaker}
  $('voice').classList.toggle('on',st.voice);applyBusy(st.busy);render(st)}catch(e){}
 setTimeout(poll,busy?400:1200)}
async function uploadFiles(list){
 for(const f of list){
  if(!f.type.startsWith('image/'))continue;
  const chip=el('div','chip up');const im=el('img');im.src=URL.createObjectURL(f);chip.appendChild(im);
  const x=el('button',null,'×');chip.appendChild(x);$('chips').appendChild(chip);
  const rec={chip,path:null};attached.push(rec);
  x.onclick=()=>{attached=attached.filter(a=>a!==rec);chip.remove()};
  try{const r=await fetch('/api/upload?name='+encodeURIComponent(f.name||'pasted.png'),{method:'POST',headers:H,body:f});const j=await r.json();
   if(j.path){rec.path=j.path;chip.classList.remove('up')}else throw 0}catch(e){attached=attached.filter(a=>a!==rec);chip.remove()}
 }}
async function send(){
 if(busy){await post('/api/stop');return}
 const t=$('inp').value.trim();const imgs=attached.filter(a=>a.path).map(a=>a.path);
 if(!t&&!imgs.length)return;
 const r=await post('/api/send',{text:t,images:imgs,voice:$('mVoice').checked,speaker:$('mSpk').value});
 if(r.error){alert(r.error);return}
 $('inp').value='';grow();attached=[];$('chips').innerHTML='';lastSig='';
 applyBusy(true);fetch('/api/state',{headers:H}).then(r=>r.json()).then(render)}
function grow(){const i=$('inp');i.style.height='auto';i.style.height=Math.min(i.scrollHeight,200)+'px'}
$('inp').addEventListener('input',grow);
$('inp').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send()}});
$('send').onclick=send;
$('inp').addEventListener('paste',e=>{const fs=[...(e.clipboardData&&e.clipboardData.files||[])].filter(f=>f.type.startsWith('image/'));if(fs.length){e.preventDefault();uploadFiles(fs)}});
const comp=$('composer');
['dragenter','dragover'].forEach(n=>window.addEventListener(n,e=>{e.preventDefault();comp.classList.add('drag')}));
['dragleave','drop'].forEach(n=>window.addEventListener(n,e=>{e.preventDefault();if(n==='drop'||e.relatedTarget===null)comp.classList.remove('drag')}));
window.addEventListener('drop',e=>{if(e.dataTransfer&&e.dataTransfer.files.length)uploadFiles([...e.dataTransfer.files])});
const menu=$('menu');
$('plus').onclick=e=>{e.stopPropagation();const r=$('plus').getBoundingClientRect();menu.style.left=r.left+'px';menu.style.bottom=(innerHeight-r.top+8)+'px';menu.style.display=menu.style.display==='block'?'none':'block'};
addEventListener('click',e=>{if(!menu.contains(e.target))menu.style.display='none'});
$('mUp').onclick=()=>{menu.style.display='none';$('file').click()};
$('file').onchange=e=>{uploadFiles([...e.target.files]);e.target.value=''};
$('mNew').onclick=async()=>{menu.style.display='none';await post('/api/reset');lastSig='';seen=-1;poll_once()};
const setVoice=async v=>{await post('/api/settings',{voice:v,speaker:$('mSpk').value})};
$('mVoice').onchange=e=>setVoice(e.target.checked);$('mSpk').onchange=()=>setVoice($('mVoice').checked);
$('voice').onclick=()=>{const v=!$('mVoice').checked;$('mVoice').checked=v;setVoice(v)};
$('big').onclick=()=>{$('big').style.display='none'};
function poll_once(){fetch('/api/state',{headers:H}).then(r=>r.json()).then(render)}
poll();
</script></body></html>)HTML";
