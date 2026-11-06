const fs=require('fs');
const path=require('node:path').join(__dirname, '../visualizations/attention-robustness.html');
const html=fs.readFileSync(path,'utf8');
const json=html.match(/<script type="application\/json" id="rb-data">([\s\S]*?)<\/script>/)[1];
const vals={regime:['penalty','no_penalty'],step:['200','600'],n:['16','64','256','1024'],difficulty:['random','similar','near_duplicate'],task:['one','two']};
const defaults={regime:'penalty',step:'600',n:'1024',difficulty:'similar',task:'two'};
const els={};
for(const [k,vs] of Object.entries(vals))els['#rb-'+k]={value:defaults[k],options:vs.map(value=>({value})),addEventListener(e,f){this[e]=f;}};
for(const k of ['bars','table'])els['#rb-'+k]={innerHTML:''};
global.document={getElementById(id){return id==='rb-data'?{textContent:json}:{querySelector(s){if(!els[s])throw Error(s);return els[s];}};}};
global.window={addEventListener(){}};
eval(html.match(/<script>\s*([\s\S]*?)<\/script>/)[1]);
let count=0;
for(const regime of vals.regime)for(const step of vals.step)for(const n of vals.n)for(const difficulty of vals.difficulty)for(const task of vals.task){
 for(const [k,v] of Object.entries({regime,step,n,difficulty,task}))els['#rb-'+k].value=v;
 els['#rb-task'].change();
 if(els['#rb-table'].innerHTML.includes('undefined')||els['#rb-table'].innerHTML.includes('NaN'))throw Error('Invalid cell');
 if(!els['#rb-bars'].innerHTML.includes('Combined'))throw Error('Missing intervention');count++;
}
console.log('Passed all '+count+' selector combinations; embedded data '+Buffer.byteLength(json)+' bytes.');
