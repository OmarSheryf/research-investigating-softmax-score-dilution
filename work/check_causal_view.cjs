const fs=require('fs');
const file=require('node:path').join(__dirname, '../visualizations/causal-checkpoints.html');
const html=fs.readFileSync(file,'utf8');
const json=html.match(/<script type="application\/json" id="cc-data">([\s\S]*?)<\/script>/)[1];
const choices={step:['400','1200'],n:['8','16','32','64'],seed:['all','11','22','33','44','55'],metric:['single','two','both_reads','mass','entropy'],dispersion:['far','near']};
const defaults={step:'1200',n:'64',seed:'all',metric:'two',dispersion:'far'};
const fields=Object.entries(choices).map(([k,values])=>({dataset:{key:k},value:defaults[k],options:values.map(value=>({value})),querySelector(){return this.options[1]},addEventListener(e,f){this[e]=f;}}));
const elements=Object.fromEntries(['status','bars','values','positions','seeds'].map(k=>['#cc-'+k,{innerHTML:'',textContent:''}]));
global.document={getElementById(id){return id==='cc-data'?{textContent:json}:{querySelectorAll(){return fields},querySelector(s){if(!elements[s])throw Error(s);return elements[s];}};}};
global.window={addEventListener(){}};
eval(html.match(/<script>\s*([\s\S]*?)<\/script>/)[1]);
let count=0;
for(const step of choices.step)for(const n of choices.n)for(const seed of choices.seed)for(const metric of choices.metric)for(const dispersion of choices.dispersion){
 for(const f of fields)f.value=({step,n,seed,metric,dispersion})[f.dataset.key];
 fields[0].change();
 for(const e of Object.values(elements))if((e.innerHTML+e.textContent).includes('NaN')||(e.innerHTML+e.textContent).includes('undefined'))throw Error('Invalid render');
 if(!elements['#cc-positions'].innerHTML.includes('Middle'))throw Error('No position detail');count++;
}
console.log('Passed '+count+' selector combinations; '+Buffer.byteLength(json)+' embedded bytes.');
