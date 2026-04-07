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
        "ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.));compressor=os.environ.get('COMPRESSOR','brotli');",
        "ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.));skip_training=bool(int(os.environ.get('SKIP_TRAINING','0')));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',-1));ngram_tilt_enabled=bool(int(os.environ.get('NGRAM_TILT_ENABLED','0')));ngram_base_beta=float(os.environ.get('NGRAM_BASE_BETA',2.));ngram_agree_bonus=float(os.environ.get('NGRAM_AGREE_BONUS',.1));ngram_within_threshold=float(os.environ.get('NGRAM_WITHIN_THRESHOLD',.25));ngram_within_beta=float(os.environ.get('NGRAM_WITHIN_BETA',0.));ngram_word_threshold=float(os.environ.get('NGRAM_WORD_THRESHOLD',.8));ngram_word_beta=float(os.environ.get('NGRAM_WORD_BETA',0.));ngram_open_table_bits=int(os.environ.get('NGRAM_OPEN_TABLE_BITS',26));ngram_order_stride=int(os.environ.get('NGRAM_ORDER_STRIDE',2));compressor=os.environ.get('COMPRESSOR','brotli');",
        "hyperparameter extension",
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
        """def eval_val_sliding_ttt(h,base_model,rank,world_size,device,val_data,stride,ngram_state=None):
\tseq_len=h.eval_seq_len;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
\tfor ws in window_starts:end=min(ws+seq_len,total_tokens);wlen=end-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
\tlog(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} freeze_blocks={h.ttt_freeze_blocks}");compiled_logits=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);frozen_block_ids=set(range(min(h.ttt_freeze_blocks,len(base_model.blocks))));ttt_params=[]
\tfor(name,p)in base_model.named_parameters():
\t\tfreeze=False
\t\tfor bi in frozen_block_ids:
\t\t\tif f"blocks.{bi}."in name:freeze=True;break
\t\tif freeze:p.requires_grad_(False)
\t\telse:p.requires_grad_(True);ttt_params.append(p)
\tlog(f"ttt_sliding:params unfrozen={sum(p.numel()for p in ttt_params)} frozen={sum(p.numel()for p in base_model.parameters()if not p.requires_grad)}");optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum);t0=time.perf_counter();batch_seqs=h.ttt_batch_seqs
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
\tif not h.skip_training:
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

    required_markers = (
        "skip_training=bool(int(os.environ.get('SKIP_TRAINING','0')))",
        "parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',-1))",
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
