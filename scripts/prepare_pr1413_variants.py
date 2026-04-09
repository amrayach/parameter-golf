#!/usr/bin/env python3
"""Materialize local PR #1413 experiment folders for A/B/C/D/E RunPod runs.

This script fetches content from local git refs `pr1413` and `pr1437`, decodes
the self-extracting `train_gpt.py` wrapper, applies only the minimal hooks
needed for the planned experiments, then rewrites a local record-style folder:

- A/C use a local mirror of upstream PR #1413 with no code changes
- B/D/E use a patched local variant with:
  - `PARALLEL_RESIDUAL_START` hook
  - `SKIP_TRAINING` hook for eval-only replay
  - causal token-only n-gram tilt support plus sidecar files
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import lzma
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_REF = "pr1413"
STACK_REF = "pr1437"

BASE_UPSTREAM_REL = Path(
    "records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828"
)
STACK_UPSTREAM_REL = Path(
    "records/track_10min_16mb/2026-04-07_SP8192_ParallelResid7_Loop35_NgramTilt"
)

BASE_LOCAL_REL = Path(
    "records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase"
)
STACK_LOCAL_REL = Path(
    "records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep"
)


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _ensure_ref(ref: str) -> None:
    subprocess.check_call(
        ["git", "rev-parse", "--verify", ref],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _git_show(ref: str, path: Path) -> str:
    return _run_git("show", f"{ref}:{path.as_posix()}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_wrapper(wrapper: str) -> str:
    match = re.search(r"b85decode\((?P<q>['\"])(.+?)(?P=q)\)", wrapper, re.S)
    if not match:
        raise ValueError("Could not locate base85 payload in wrapped train_gpt.py")
    encoded = match.group(2).encode("utf-8")
    return lzma.decompress(
        base64.b85decode(encoded),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    ).decode("utf-8")


def _wrap_source(source: str) -> str:
    """Re-wrap patched source using the same format as upstream PR #1413.

    Upstream wrappers use raw LZMA2 (FORMAT_RAW + FILTER_LZMA2) and pass the
    base85-encoded payload as an ASCII string literal, not bytes.  The exec line
    therefore must include the matching decompressor arguments.
    """
    compressed = lzma.compress(
        source.encode("utf-8"),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}],
    )
    encoded = base64.b85encode(compressed).decode("ascii")
    return (
        "import lzma as L,base64 as B\n"
        f'exec(L.decompress(B.b85decode("{encoded}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))\n'
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"Missing expected {label} snippet")
    return text.replace(old, new, 1)


def _replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not locate {label} block")
    return text[: match.start()] + replacement + text[match.end() :]


def _patch_stack_source(pr1413_source: str) -> str:
    source = pr1413_source

    source = _replace_once(
        source,
        "import collections,copy,glob,io,lzma,math,os",
        "import collections,copy,glob,io,json,lzma,math,os,struct",
        "custom pack imports",
    )

    source = _replace_once(
        source,
        "ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.));compressor=os.environ.get('COMPRESSOR','brotli');",
        "ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.));ttt_optimizer=os.environ.get('TTT_OPTIMIZER','sgd');ttt_decay=float(os.environ.get('TTT_DECAY',0.));skip_training=bool(int(os.environ.get('SKIP_TRAINING','0')));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',-1));cautious_muon=bool(int(os.environ.get('CAUTIOUS_MUON','0')));owc_enabled=bool(int(os.environ.get('OWC_ENABLED','0')));owc_gamma_steps=int(os.environ.get('OWC_GAMMA_STEPS',10));owc_scope=os.environ.get('OWC_SCOPE','all');cdquant_enabled=bool(int(os.environ.get('CDQUANT_ENABLED','0')));cdquant_iters=int(os.environ.get('CDQUANT_ITERS',3));ngram_tilt_enabled=bool(int(os.environ.get('NGRAM_TILT_ENABLED','0')));ngram_base_beta=float(os.environ.get('NGRAM_BASE_BETA',2.));ngram_agree_bonus=float(os.environ.get('NGRAM_AGREE_BONUS',.1));ngram_within_threshold=float(os.environ.get('NGRAM_WITHIN_THRESHOLD',.25));ngram_within_beta=float(os.environ.get('NGRAM_WITHIN_BETA',0.));ngram_word_threshold=float(os.environ.get('NGRAM_WORD_THRESHOLD',.8));ngram_word_beta=float(os.environ.get('NGRAM_WORD_BETA',0.));ngram_open_table_bits=int(os.environ.get('NGRAM_OPEN_TABLE_BITS',26));ngram_order_stride=int(os.environ.get('NGRAM_ORDER_STRIDE',2));export_packing=os.environ.get('EXPORT_PACKING','torchsave');requant_only=bool(int(os.environ.get('REQUANT_ONLY','0')));compressor=os.environ.get('COMPRESSOR','brotli');",
        "hyperparameter extension",
    )

    # --- Cautious Muon: pass flag through param_groups, mask before Newton-Schulz ---
    source = _replace_once(
        source,
        "def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))",
        "def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False,cautious=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize,cautious=cautious))",
        "cautious muon init",
    )
    source = _replace_once(
        source,
        (
            "\t\t\t\t\tif nesterov:g=g.add(buf,alpha=momentum)\n"
            "\t\t\t\t\tif group.get('row_normalize',False):"
        ),
        (
            "\t\t\t\t\tif nesterov:g=g.add(buf,alpha=momentum)\n"
            "\t\t\t\t\tif group.get('cautious',False):mask=(buf.sign()==p.grad.sign()).float();g=g*mask\n"
            "\t\t\t\t\tif group.get('row_normalize',False):"
        ),
        "cautious muon mask",
    )
    source = _replace_once(
        source,
        "self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)",
        "self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize,cautious=h.cautious_muon)",
        "cautious muon instantiation",
    )

    # --- OWC: inject _owc_optimize_clip_sigmas function before gptq_quantize_weight ---
    source = _replace_once(
        source,
        "def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):",
        (
            "def _owc_optimize_clip_sigmas(W,H,clip_range,initial_sigmas,n_steps,row_std):\n"
            "\trows,cols=W.shape;W_f=W.float().detach();H_f=H.float().detach();rs=row_std.float().detach()\n"
            "\tlog_gamma=torch.full((rows,),math.log(max(initial_sigmas,1e-6)),dtype=torch.float32,device=W.device,requires_grad=True);opt=torch.optim.Adam([log_gamma],lr=0.05)\n"
            "\tfor _ in range(n_steps):\n"
            "\t\topt.zero_grad();gamma=log_gamma.exp();s=(gamma*rs/clip_range).clamp_min(1e-10);sf=s.unsqueeze(1)\n"
            "\t\tq=torch.clamp(torch.round(W_f/sf),-clip_range,clip_range).detach();W_hat=q*sf;err=W_f-W_hat\n"
            "\t\tloss=((err@H_f)*err).sum();loss.backward();opt.step()\n"
            "\treturn log_gamma.exp().detach()\n"
            "def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128,owc_steps=0,cdquant_iters=0):"
        ),
        "OWC function + GPTQ signature extension",
    )

    # --- OWC + CDQuant: modify GPTQ body to use OWC clip sigmas and CDQuant rounding ---
    source = _replace_once(
        source,
        (
            "\tW_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();"
            "dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);"
            "perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);"
            "W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];"
            "Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));"
            "Hinv=torch.linalg.cholesky(Hinv,upper=True);"
            "row_std=W_orig.std(dim=1);"
            "s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);"
            "sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()"
        ),
        (
            "\tW_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();"
            "dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);"
            "row_std=W_orig.std(dim=1)\n"
            "\tif owc_steps>0:opt_sigmas=_owc_optimize_clip_sigmas(W_orig,H,clip_range,clip_sigmas,owc_steps,row_std);s=(opt_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16)\n"
            "\telse:s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16)\n"
            "\tsf=s.float();perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);"
            "W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];"
            "Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));"
            "Hinv=torch.linalg.cholesky(Hinv,upper=True);"
            "Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()"
        ),
        "OWC clip sigma + CDQuant prep in GPTQ body",
    )

    # --- CDQuant: Hessian-weighted floor/ceil rounding in GPTQ inner loop ---
    # Uses Hinv diagonal to weight reconstruction error: pick floor or ceil per
    # element based on which produces lower Hessian-weighted squared error for
    # that column.  This differs from torch.round() when the Hessian diagonal
    # varies across columns (high-sensitivity columns get more careful rounding).
    source = _replace_once(
        source,
        (
            "\t\tfor j in range(i2-i1):"
            "w_col=W_block[:,j];d=Hinv_block[j,j];"
            "q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);"
            "Q[:,i1+j]=q_col.to(torch.int8);"
            "err=(w_col-q_col.float()*sf)/d;"
            "Err[:,j]=err;"
            "W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)"
        ),
        (
            "\t\tfor j in range(i2-i1):\n"
            "\t\t\tw_col=W_block[:,j];d=Hinv_block[j,j];scaled=w_col/sf\n"
            "\t\t\tif cdquant_iters>0:\n"
            "\t\t\t\tfl=torch.clamp(torch.floor(scaled),-clip_range,clip_range);ce=torch.clamp(fl+1,-clip_range,clip_range);err_fl=(w_col-fl*sf);err_ce=(w_col-ce*sf);cost_fl=err_fl*err_fl*d;cost_ce=err_ce*err_ce*d;q_col=torch.where(cost_ce<cost_fl,ce,fl)\n"
            "\t\t\telse:q_col=torch.clamp(torch.round(scaled),-clip_range,clip_range)\n"
            "\t\t\tQ[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)"
        ),
        "CDQuant rounding in GPTQ inner loop",
    )

    # --- OWC + CDQuant: update gptq_mixed_quantize call site ---
    source = _replace_once(
        source,
        (
            "\t\tcs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;"
            "bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;"
            "q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);"
            "result[name+'.q']=q;result[name+'.scale']=s;"
            'meta[name]=f"gptq (int{bits})"'
        ),
        (
            "\t\tcs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;"
            "bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;"
            "cat='embed' if'tok_emb'in name else('attn' if'.attn.'in name else('mlp' if'.mlp.'in name else'other'));"
            "owc_allowed=(h.owc_scope=='all')or(h.owc_scope=='matrix' and cat in('attn','mlp'))or(h.owc_scope==cat);"
            "owc=h.owc_gamma_steps if h.owc_enabled and owc_allowed else 0;"
            "cdi=h.cdquant_iters if h.cdquant_enabled else 0;"
            "q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1,owc_steps=owc,cdquant_iters=cdi);"
            "result[name+'.q']=q;result[name+'.scale']=s;"
            'meta[name]=f"gptq (int{bits})"+(f" owc{owc}" if owc else "")+(f" cd{cdi}" if cdi else "")'
        ),
        "OWC+CDQuant call site in gptq_mixed_quantize",
    )

    source = _replace_once(
        source,
        'for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")',
        'for cat in sorted(categories):log(f"  {cat}: {\', \'.join(sorted(categories[cat]))}")',
        "quantized weights summary f-string fix",
    )

    source = _replace_once(
        source,
        (
            "class Block(nn.Module):\n"
            "\tdef __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.\n"
            "\tdef forward(self,x,x0):mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor);return x_out\n"
        ),
        (
            "class Block(nn.Module):\n"
            "\tdef __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.parallel=False\n"
            "\tdef forward(self,x,x0):\n"
            "\t\tmix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor)\n"
            "\t\tif self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out\n"
            "\t\telse:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)\n"
            "\t\treturn x_out\n"
        ),
        "parallel residual block",
    )

    source = _replace_once(
        source,
        (
            "\t\tif h.xsa_last_n>0:\n"
            "\t\t\tfor i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True\n"
            "\t\tself.looping_active=False\n"
        ),
        (
            "\t\tif h.xsa_last_n>0:\n"
            "\t\t\tfor i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True\n"
            "\t\tif h.parallel_residual_start>=0:\n"
            "\t\t\tfor i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=True\n"
            "\t\tself.looping_active=False\n"
        ),
        "parallel residual enable hook",
    )

    source = _replace_between(
        source,
        "def eval_val_sliding_ttt(h,base_model,rank,world_size,device,val_data,stride):",
        "def timed_eval(label,fn,*args,**kwargs):",
        """class RMSDecay(torch.optim.Optimizer):
\t"RMSProp + decay toward pre-TTT anchor (Krause 2018 dynamic evaluation)."
\tdef __init__(self,params,anchors,lr=0.005,alpha=0.9,eps=1e-8,decay=0.001):
\t\tdefaults=dict(lr=lr,alpha=alpha,eps=eps,decay=decay);params_list=list(params);super().__init__(params_list,defaults)
\t\tfor group in self.param_groups:
\t\t\tfor i,p in enumerate(group['params']):self.state[p]['anchor']=anchors[i]
\t@torch.no_grad()
\tdef step(self):
\t\tfor group in self.param_groups:
\t\t\tlr=group['lr'];alpha=group['alpha'];eps=group['eps'];decay=group['decay']
\t\t\tfor p in group['params']:
\t\t\t\tif p.grad is None:continue
\t\t\t\tstate=self.state[p]
\t\t\t\tif 'v' not in state:state['v']=torch.zeros_like(p)
\t\t\t\tv=state['v'];g=p.grad;v.mul_(alpha).addcmul_(g,g,value=1-alpha);p.addcdiv_(g,v.sqrt().add_(eps),value=-lr)
\t\t\t\tif decay>0:p.add_(state['anchor']-p,alpha=decay)
def eval_val_sliding_ttt(h,base_model,rank,world_size,device,val_data,stride,ngram_state=None):
\tseq_len=h.eval_seq_len;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
\tfor ws in window_starts:end=min(ws+seq_len,total_tokens);wlen=end-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
\tlog(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} freeze_blocks={h.ttt_freeze_blocks}");compiled_logits=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);frozen_block_ids=set(range(min(h.ttt_freeze_blocks,len(base_model.blocks))));ttt_params=[]
\tfor(name,p)in base_model.named_parameters():
\t\tfreeze=False
\t\tfor bi in frozen_block_ids:
\t\t\tif f"blocks.{bi}."in name:freeze=True;break
\t\tif freeze:p.requires_grad_(False)
\t\telse:p.requires_grad_(True);ttt_params.append(p)
\tlog(f"ttt_sliding:params unfrozen={sum(p.numel()for p in ttt_params)} frozen={sum(p.numel()for p in base_model.parameters()if not p.requires_grad)}")
\tif h.ttt_optimizer=='rmsdecay':ttt_anchors=[p.detach().clone()for p in ttt_params];optimizer=RMSDecay(ttt_params,ttt_anchors,lr=h.ttt_lr,alpha=h.ttt_momentum,decay=h.ttt_decay);log(f"ttt_sliding:optimizer=RMSDecay alpha={h.ttt_momentum} decay={h.ttt_decay}")
\telse:optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum);log(f"ttt_sliding:optimizer=SGD momentum={h.ttt_momentum}")
\tt0=time.perf_counter();batch_seqs=h.ttt_batch_seqs
\tfor ci in range(num_chunks):
\t\twindows=chunk_windows[ci]
\t\tif not windows:continue
\t\tchunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
\t\twith torch.no_grad():
\t\t\tfor bi in range(0,len(my_windows),batch_seqs):
\t\t\t\tbatch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
\t\t\t\tfor(i,ws)in enumerate(batch_ws):end=min(ws+seq_len,total_tokens);wlen=end-ws;wlens.append(wlen);chunk_tok=val_data.val_tokens[ws:end+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
\t\t\t\twith torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=compiled_logits(x_batch)
\t\t\t\tlogits_f=logits.float();nll=F.cross_entropy(logits_f.reshape(-1,logits_f.size(-1)),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
\t\t\t\tfor(i,ws)in enumerate(batch_ws):
\t\t\t\t\twlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64)
\t\t\t\t\tif ngram_state is not None and wlen-s>0:gp=torch.arange(ws+s+1,ws+wlen+1,device=device,dtype=torch.int64);scored_nll=ngram_state.tilt_nll(scored_nll=scored_nll,scored_logits=logits_f[i,s:wlen],target_ids=y_batch[i,s:wlen],global_positions=gp)
\t\t\t\t\tloss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
\t\tis_last_chunk=ci==num_chunks-1
\t\tif not is_last_chunk and h.ttt_epochs>0:
\t\t\tbase_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
\t\t\tif chunk_seqs>0:
\t\t\t\tcos_lr=h.ttt_lr*.5*(1.+math.cos(math.pi*ci/max(num_chunks-1,1)))
\t\t\t\tfor pg in optimizer.param_groups:pg['lr']=cos_lr
\t\t\t\tmy_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
\t\t\t\tfor _ep in range(h.ttt_epochs):
\t\t\t\t\tfor bs in range(0,my_chunk_seqs,batch_seqs):
\t\t\t\t\t\tbe=min(bs+batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
\t\t\t\t\t\tif end_tok>val_data.val_tokens.numel():continue
\t\t\t\t\t\tlocal=val_data.val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=True)
\t\t\t\t\t\twith torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=base_model(x,y)
\t\t\t\t\t\tloss.backward()
\t\t\t\t\t\tif world_size>1:
\t\t\t\t\t\t\tfor p in ttt_params:
\t\t\t\t\t\t\t\tif p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
\t\t\t\t\t\ttorch.nn.utils.clip_grad_norm_(ttt_params,h.ttt_grad_clip);optimizer.step()
\t\tif rank==0 and(ci%10==0 or ci==num_chunks-1):elapsed=time.perf_counter()-t0;rl=loss_sum.item()/max(token_count.item(),1);rbpb=rl/math.log(2.)*(token_count.item()/max(byte_count.item(),1));log(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
\tif dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
\tval_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item())
\tfor p in base_model.parameters():p.requires_grad_(True)
\tbase_model.eval();log(f"ttt_sliding:done val_loss={val_loss:.6f}{ val_bpb=:.6f} elapsed={time.perf_counter()-t0:.1f}s");return val_loss,val_bpb
def timed_eval(label,fn,*args,**kwargs):""",
        "eval_val_sliding_ttt block",
    )

    source = _replace_between(
        source,
        "def train_and_eval(h,device):",
        "def main():",
        """def train_and_eval(h,device):
\trandom.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}")
\tif h.requant_only:
\t\tlog(f'requant_only: loading {h.model_path} for re-quantization with owc_enabled={h.owc_enabled} owc_scope={h.owc_scope} owc_gamma_steps={h.owc_gamma_steps}');base_model=GPT(h).to(device);base_model.load_state_dict(torch.load(h.model_path,map_location=device,weights_only=True));serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))
\telif not h.skip_training:
\t\tbase_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))
\telse:
\t\tlog('skip_training:reusing existing quantized artifact')
\t\tif not Path(h.quantized_model_path).exists():raise FileNotFoundError(f"Missing quantized model for eval-only run: {h.quantized_model_path}")
\tif h.distributed:dist.barrier()
\teval_model=deserialize(h,device)
\tif h.num_loops>0:eval_model.looping_active=True
\tcompiled_model=torch.compile(eval_model,dynamic=False,fullgraph=True);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
\tif h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
\tif h.ttt_enabled:
\t\tdel eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)
\t\tif h.num_loops>0:ttt_model.looping_active=True
\t\tngram_state=None
\t\tif h.ngram_tilt_enabled:
\t\t\tfrom ngram_tilt import NgramTiltState
\t\t\tngram_state=NgramTiltState(val_tokens=val_data.val_tokens,has_leading_space_lut=val_data.has_leading_space_lut,is_boundary_token_lut=val_data.is_boundary_token_lut,rank=h.rank,world_size=h.world_size,device=device,base_beta=h.ngram_base_beta,agree_bonus=h.ngram_agree_bonus,within_threshold=h.ngram_within_threshold,within_beta=h.ngram_within_beta,word_threshold=h.ngram_word_threshold,word_beta=h.ngram_word_beta,open_table_bits=h.ngram_open_table_bits,order_stride=h.ngram_order_stride,log=log)
\t\ttimed_eval('legal_ttt_exact',eval_val_sliding_ttt,h,ttt_model,h.rank,h.world_size,device,val_data,stride=h.eval_stride,ngram_state=ngram_state);del ttt_model
\t\tif ngram_state is not None:del ngram_state;torch.cuda.empty_cache()
def main():""",
        "train_and_eval block",
    )

    source = _replace_once(
        source,
        'log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}")',
        'log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob(\'fineweb_train_*.bin\')))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}")',
        "train shard logging quote fix",
    )

    source = _replace_once(
        source,
        "def _compress(data,compressor):",
        """_DTYPE_TO_STR={torch.float32:'f32',torch.float16:'f16',torch.bfloat16:'bf16',torch.int8:'i8',torch.int16:'i16',torch.int32:'i32',torch.int64:'i64',torch.bool:'bool'}
_STR_TO_NP={'f32':np.float32,'f16':np.float16,'bf16':np.float32,'i8':np.int8,'i16':np.int16,'i32':np.int32,'i64':np.int64,'bool':np.bool_}
_DTYPE_ELEM_SIZE={'f32':4,'f16':2,'bf16':2,'i8':1,'i16':2,'i32':4,'i64':8,'bool':1}
def _custom_pack(state_dict,meta,shuffle=True):
\theader_entries={};chunks=[];offset=0
\tfor name in sorted(state_dict.keys()):
\t\ttensor=state_dict[name];dtype_str=_DTYPE_TO_STR.get(tensor.dtype)
\t\tif dtype_str is None:raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")
\t\traw=tensor.float().numpy().tobytes()if tensor.dtype==torch.bfloat16 else tensor.numpy().tobytes();elem_size=_DTYPE_ELEM_SIZE[dtype_str]
\t\tif shuffle and elem_size>1:raw=_byte_shuffle(raw,elem_size)
\t\theader_entries[name]={'s':list(tensor.shape),'d':dtype_str,'o':offset,'n':len(raw),'e':tensor.numel()}
\t\tchunks.append(raw);offset+=len(raw)
\theader_json=json.dumps({'t':header_entries,'m':meta},separators=(',',':')).encode()
\treturn struct.pack('<I',len(header_json))+header_json+b''.join(chunks)
def _custom_unpack(blob,shuffle=True):
\theader_len=struct.unpack('<I',blob[:4])[0];header=json.loads(blob[4:4+header_len]);data_start=4+header_len;state_dict={}
\tfor(name,info)in header['t'].items():
\t\traw=blob[data_start+info['o']:data_start+info['o']+info['n']];dtype_str=info['d'];elem_size=_DTYPE_ELEM_SIZE[dtype_str]
\t\tif shuffle and elem_size>1:raw=_byte_unshuffle(raw)
\t\tif dtype_str=='bf16':tensor=torch.from_numpy(np.frombuffer(bytearray(raw),dtype=np.float32).copy()).to(torch.bfloat16).reshape(info['s'])
\t\telse:tensor=torch.from_numpy(np.frombuffer(bytearray(raw),dtype=_STR_TO_NP[dtype_str]).copy()).reshape(info['s'])
\t\tstate_dict[name]=tensor
\treturn state_dict,header['m']
def _compress(data,compressor,already_shuffled=False):""",
        "custom pack helpers",
    )

    source = _replace_once(
        source,
        "\tdata=_byte_shuffle(data)\n\tif compressor=='lzma':return lzma.compress(data,preset=6)\n\telif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)\n\traise ValueError(f\"Unknown compressor: {compressor!r}\")\ndef _decompress(data,compressor):\n\tif compressor=='lzma':raw=lzma.decompress(data)\n\telif compressor=='brotli':import brotli;raw=brotli.decompress(data)\n\telse:raise ValueError(f\"Unknown compressor: {compressor!r}\")\n\traw=_byte_unshuffle(raw);return raw",
        "\tif not already_shuffled:data=_byte_shuffle(data)\n\tif compressor=='lzma':return lzma.compress(data,preset=6)\n\telif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)\n\traise ValueError(f\"Unknown compressor: {compressor!r}\")\ndef _decompress(data,compressor,already_shuffled=False):\n\tif compressor=='lzma':raw=lzma.decompress(data)\n\telif compressor=='brotli':import brotli;raw=brotli.decompress(data)\n\telse:raise ValueError(f\"Unknown compressor: {compressor!r}\")\n\tif not already_shuffled:raw=_byte_unshuffle(raw)\n\treturn raw",
        "custom pack compressor branch",
    )

    source = _replace_once(
        source,
        "quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes",
        "quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_raw=_custom_pack(quant_result,quant_meta,shuffle=True)if h.export_packing=='custompack' else io.BytesIO();\n\tif h.export_packing!='custompack':torch.save({'w':quant_result,'m':quant_meta},quant_raw);quant_raw=quant_raw.getvalue()\n\tquant_blob=_compress(quant_raw,h.compressor,already_shuffled=h.export_packing=='custompack');quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes",
        "serialize export branch",
    )

    source = _replace_once(
        source,
        "quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model",
        "quant_raw=_decompress(quant_blob_disk,h.compressor,already_shuffled=h.export_packing=='custompack')\n\tif h.export_packing=='custompack':quant_w,quant_m=_custom_unpack(quant_raw,shuffle=True)\n\telse:quant_state=torch.load(io.BytesIO(quant_raw),map_location='cpu');quant_w,quant_m=quant_state['w'],quant_state['m']\n\tdeq_state=dequantize_mixed(quant_w,quant_m,sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model",
        "deserialize export branch",
    )

    required_markers = (
        "skip_training=bool(int(os.environ.get('SKIP_TRAINING','0')))",
        "parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',-1))",
        "cautious_muon=bool(int(os.environ.get('CAUTIOUS_MUON','0')))",
        "owc_enabled=bool(int(os.environ.get('OWC_ENABLED','0')))",
        "owc_scope=os.environ.get('OWC_SCOPE','all')",
        "cdquant_enabled=bool(int(os.environ.get('CDQUANT_ENABLED','0')))",
        "export_packing=os.environ.get('EXPORT_PACKING','torchsave')",
        "requant_only=bool(int(os.environ.get('REQUANT_ONLY','0')))",
        "requant_only: loading",
        "def _custom_pack(state_dict,meta,shuffle=True):",
        "def _owc_optimize_clip_sigmas(",
        "if cdquant_iters>0:",
        "owc_allowed=(h.owc_scope=='all')or(h.owc_scope=='matrix' and cat in('attn','mlp'))or(h.owc_scope==cat)",
        "if group.get('cautious',False):mask=",
        "def eval_val_sliding_ttt(h,base_model,rank,world_size,device,val_data,stride,ngram_state=None):",
        "if h.ngram_tilt_enabled:",
        "skip_training:reusing existing quantized artifact",
    )
    for marker in required_markers:
        if marker not in source:
            raise ValueError(f"Patched source is missing expected marker: {marker}")
    return source


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_base_readme(base_readme: str, manifest: dict) -> str:
    return f"""# Local Mirror: PR #1413 faithful base

This folder is a local mirror of upstream `{BASE_REF}:{BASE_UPSTREAM_REL.as_posix()}` for
offline RunPod batch execution. It exists so A/C can run from the local repo without
an on-pod fetch.

- local record folder: `{BASE_LOCAL_REL.as_posix()}`
- wrapped code bytes: `{manifest['wrapper_bytes']}`
- decoded source sha256: `{manifest['decoded_source_sha256']}`

Use:

```bash
FETCH_PAYLOAD=0 RECORD_REL={BASE_LOCAL_REL.as_posix()} bash scripts/runpod_1413.sh 0
```

Upstream README follows.

---

{base_readme}
"""


def _write_stack_readme(manifest: dict) -> str:
    return f"""# Local Prep: PR #1413 parallel-residual and n-gram variant

This folder is a local, non-submission prep variant built from upstream `{BASE_REF}` and
the corrected causal n-gram sidecars from `{STACK_REF}`. It is designed for the planned
A/B/C/D/E RunPod batch.

What is added on top of faithful `#1413`:

- `PARALLEL_RESIDUAL_START` hook, default `-1` (disabled)
- `SKIP_TRAINING` hook, default `0`
- causal token-only n-gram tilt support, default disabled
- sidecars `ngram_tilt.py` and `fused_expert_kernel.cpp`

Intended runs:

- `B`: `PARALLEL_RESIDUAL_START=7`
- `D`: `PARALLEL_RESIDUAL_START=7 LOOP_START=3`
- `E`: reuse the `D` checkpoint with `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1`

Important defaults for corrected token-only n-gram tilt:

- `NGRAM_BASE_BETA=2.0`
- `NGRAM_AGREE_BONUS=0.1`
- `NGRAM_WITHIN_THRESHOLD=0.25`
- `NGRAM_WITHIN_BETA=0.0`
- `NGRAM_WORD_THRESHOLD=0.8`
- `NGRAM_WORD_BETA=0.0`
- `NGRAM_OPEN_TABLE_BITS=26`
- `NGRAM_ORDER_STRIDE=2`

Materialization summary:

- local record folder: `{STACK_LOCAL_REL.as_posix()}`
- wrapped code bytes: `{manifest['wrapper_bytes']}`
- decoded source sha256: `{manifest['decoded_source_sha256']}`
- sidecar sha256:
  - `ngram_tilt.py`: `{manifest['ngram_tilt_sha256']}`
  - `fused_expert_kernel.cpp`: `{manifest['fused_kernel_sha256']}`
"""


def _write_variant_folder(
    out_dir: Path,
    readme: str,
    submission: dict,
    wrapped_train_script: str,
    manifest: dict,
    *,
    ngram_tilt: str | None = None,
    fused_kernel: str | None = None,
    force: bool,
) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} already exists; re-run with --force")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_json(out_dir / "submission.json", submission)
    (out_dir / "train_gpt.py").write_text(wrapped_train_script, encoding="utf-8")
    _write_json(out_dir / "variant_manifest.json", manifest)
    if ngram_tilt is not None:
        (out_dir / "ngram_tilt.py").write_text(ngram_tilt, encoding="utf-8")
    if fused_kernel is not None:
        (out_dir / "fused_expert_kernel.cpp").write_text(fused_kernel, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local variant folders if they already exist.",
    )
    args = parser.parse_args()

    _ensure_ref(BASE_REF)
    _ensure_ref(STACK_REF)

    base_readme = _git_show(BASE_REF, BASE_UPSTREAM_REL / "README.md")
    base_submission = json.loads(_git_show(BASE_REF, BASE_UPSTREAM_REL / "submission.json"))
    base_wrapper = _git_show(BASE_REF, BASE_UPSTREAM_REL / "train_gpt.py")
    base_source = _decode_wrapper(base_wrapper)

    stack_source = _patch_stack_source(base_source)
    stack_wrapper = _wrap_source(stack_source)

    # Verify the wrapper is faithful: decode it back and compare.
    stack_roundtrip = _decode_wrapper(stack_wrapper)
    if stack_roundtrip != stack_source:
        raise ValueError(
            "Stack wrapper roundtrip failed: decoded source does not match patched source.\n"
            f"  patched len={len(stack_source)}, decoded len={len(stack_roundtrip)}"
        )

    ngram_tilt = _git_show(STACK_REF, STACK_UPSTREAM_REL / "ngram_tilt.py")
    fused_kernel = _git_show(STACK_REF, STACK_UPSTREAM_REL / "fused_expert_kernel.cpp")
    ast.parse(ngram_tilt)

    prep_date = str(date.today())

    base_local_submission = dict(base_submission)
    base_local_submission["name"] = "Local mirror: PR #1413 faithful base"
    base_local_submission["blurb"] = (
        "Local mirror of upstream PR #1413 record folder for offline A/C RunPod runs. "
        "Metrics remain the upstream reference values; this folder exists for reproducible "
        "local execution without an on-pod fetch."
    )
    base_local_submission["date"] = prep_date

    stack_submission = dict(base_submission)
    stack_submission["name"] = "Local prep: PR #1413 parallel residual and n-gram variants"
    stack_submission["blurb"] = (
        "Locally prepared, non-submission experiment folder derived from PR #1413 with "
        "dormant parallel-residual, skip-training, and corrected causal token-only "
        "n-gram tilt hooks for B/D/E ablations."
    )
    stack_submission["date"] = prep_date

    base_manifest = {
        "base_ref": BASE_REF,
        "decoded_source_sha256": _sha256_text(base_source),
        "local_record_rel": BASE_LOCAL_REL.as_posix(),
        "source_record_rel": BASE_UPSTREAM_REL.as_posix(),
        "variant": "faithful_base_local_mirror",
        "wrapper_bytes": len(base_wrapper.encode("utf-8")),
    }

    stack_manifest = {
        "base_ref": BASE_REF,
        "base_source_sha256": _sha256_text(base_source),
        "decoded_source_sha256": _sha256_text(stack_source),
        "default_ngram_config": {
            "NGRAM_AGREE_BONUS": 0.1,
            "NGRAM_BASE_BETA": 2.0,
            "NGRAM_OPEN_TABLE_BITS": 26,
            "NGRAM_ORDER_STRIDE": 2,
            "NGRAM_WITHIN_BETA": 0.0,
            "NGRAM_WITHIN_THRESHOLD": 0.25,
            "NGRAM_WORD_BETA": 0.0,
            "NGRAM_WORD_THRESHOLD": 0.8,
        },
        "default_parallel_residual_start": 7,
        "default_skip_training": 0,
        "fused_kernel_sha256": _sha256_text(fused_kernel),
        "local_record_rel": STACK_LOCAL_REL.as_posix(),
        "ngram_tilt_sha256": _sha256_text(ngram_tilt),
        "runs": {
            "A": {
                "record_rel": BASE_LOCAL_REL.as_posix(),
                "env": {},
            },
            "B": {
                "record_rel": STACK_LOCAL_REL.as_posix(),
                "env": {"PARALLEL_RESIDUAL_START": 7},
            },
            "C": {
                "record_rel": BASE_LOCAL_REL.as_posix(),
                "env": {"LOOP_END": 5, "LOOP_START": 3},
            },
            "D": {
                "record_rel": STACK_LOCAL_REL.as_posix(),
                "env": {"LOOP_END": 5, "LOOP_START": 3, "PARALLEL_RESIDUAL_START": 7},
            },
            "E": {
                "record_rel": STACK_LOCAL_REL.as_posix(),
                "env": {
                    "LOOP_END": 5,
                    "LOOP_START": 3,
                    "NGRAM_AGREE_BONUS": 0.1,
                    "NGRAM_BASE_BETA": 2.0,
                    "NGRAM_OPEN_TABLE_BITS": 26,
                    "NGRAM_ORDER_STRIDE": 2,
                    "NGRAM_TILT_ENABLED": 1,
                    "NGRAM_WITHIN_BETA": 0.0,
                    "NGRAM_WITHIN_THRESHOLD": 0.25,
                    "NGRAM_WORD_BETA": 0.0,
                    "NGRAM_WORD_THRESHOLD": 0.8,
                    "PARALLEL_RESIDUAL_START": 7,
                    "SKIP_TRAINING": 1,
                },
                "note": "Requires an existing final_model.int6.ptz in the stack folder, typically from run D.",
                "paid_session_followup": "R1 only: sweep NGRAM_WITHIN_BETA / NGRAM_WORD_BETA for potential free BPB. Keep local legality-audit defaults unchanged until the paid run.",
            },
        },
        "source_record_rel": BASE_UPSTREAM_REL.as_posix(),
        "stack_ref": STACK_REF,
        "stack_source_record_rel": STACK_UPSTREAM_REL.as_posix(),
        "variant": "parallel_residual_plus_corrected_token_only_ngram_prep",
        "wrapper_bytes": len(stack_wrapper.encode("utf-8")),
    }

    _write_variant_folder(
        REPO_ROOT / BASE_LOCAL_REL,
        _write_base_readme(base_readme, base_manifest),
        base_local_submission,
        base_wrapper,
        base_manifest,
        force=args.force,
    )
    _write_variant_folder(
        REPO_ROOT / STACK_LOCAL_REL,
        _write_stack_readme(stack_manifest),
        stack_submission,
        stack_wrapper,
        stack_manifest,
        ngram_tilt=ngram_tilt,
        fused_kernel=fused_kernel,
        force=args.force,
    )

    print("Prepared local PR #1413 experiment folders:")
    print(f"  base : {BASE_LOCAL_REL} ({base_manifest['wrapper_bytes']} code bytes)")
    print(f"  stack: {STACK_LOCAL_REL} ({stack_manifest['wrapper_bytes']} code bytes)")


if __name__ == "__main__":
    main()
