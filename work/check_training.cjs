const fs=require('fs');
const html=fs.readFileSync(require('node:path').join(__dirname, '../visualizations/attention-training.html'),'utf8');
const json=html.match(/<script type="application\/json" id="tl-data">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(json);
const elements={};
for(const id of ['tl-checkpoint','tl-length','tl-play','tl-step','tl-status','tl-metrics','tl-position','tl-record'])elements['#'+id]={value:id==='tl-checkpoint'?'8':'256',textContent:'',innerHTML:'',addEventListener(e,f){this[e]=f;}};
global.document={getElementById(id){return id==='tl-data'?{textContent:json}:{querySelector(s){if(!elements[s])throw Error(s);return elements[s];}};}};
global.window={addEventListener(){}};
eval(html.match(/<script>\s*([\s\S]*?)<\/script>/)[1]);
if(!elements['#tl-record'].innerHTML.includes('400'))throw Error('Missing checkpoints');
elements['#tl-checkpoint'].value='0';elements['#tl-checkpoint'].input();
if(elements['#tl-record'].innerHTML.includes('400'))throw Error('Slider failed');
elements['#tl-length'].value='64';elements['#tl-length'].change();
if(!elements['#tl-position'].innerHTML.includes('Middle'))throw Error('Position render failed');
for(const r of data.runs){
 if(r.checkpoints.length!==9)throw Error('Incomplete training');
 if(r.checkpoints[8].metrics[0].loss>=r.checkpoints[0].metrics[0].loss)throw Error('No learning');
 for(const c of r.checkpoints)for(const m of c.metrics)for(const k of ['accuracy','retrieval','mass','entropy'])if(!Number.isFinite(m[k])||m[k]<0||m[k]>1)throw Error('Invalid metric');
}
console.log('Passed: checkpoint slider, evaluation-length interaction, position rendering, finite metrics and training loss reduction in all variants.');
