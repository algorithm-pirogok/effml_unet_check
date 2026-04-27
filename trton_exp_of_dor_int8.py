# ============================================================
# Triton int8 tl.dot — Colab diagnostic                                                                                                                                                                                   
# Identifies which Triton-internal assertion rejects int8 dot
# and at which tile sizes (vs fp16 baseline).                                                                                                                                                                             
# ============================================================                                                                                                                                                            
import os, traceback, textwrap                                                                                                                                                                                            
import torch, triton, triton.language as tl                                                                                                                                                                               
                                                                                                                                                                                                                        
# ---------- environment banner -----------------------------------------                                                                                                                                                 
p = torch.cuda.get_device_properties(0)                                                                                                                                                                                   
print("=" * 72)                                                                                                                                                                                                           
print(f"triton  : {triton.__version__}")
print(f"torch   : {torch.__version__}")                                                                                                                                                                                   
print(f"GPU     : {p.name}  cc={p.major}.{p.minor}")                                                                                                                                                                      
print(f"smem    : {p.shared_memory_per_block_optin} B/block (opt-in)")
print(f"path    : {os.path.dirname(triton.__file__)}")                                                                                                                                                                    
print("=" * 72)                                                                                                                                                                                                           
                                                                                                                                                                                                                        
# ---------- helpers ----------------------------------------------------                                                                                                                                                 
def first_triton_internal_frame(exc):
    """Return (file, lineno, funcname, source_line) for the deepest frame                                                                                                                                                 
    that lives inside the installed triton package (not user code)."""                                                                                                                                                    
    triton_root = os.path.dirname(triton.__file__)                                                                                                                                                                        
    for f in reversed(traceback.extract_tb(exc.__traceback__)):                                                                                                                                                           
        if (f.filename or "").startswith(triton_root):
            rel = os.path.relpath(f.filename, os.path.dirname(triton_root))                                                                                                                                               
            return rel, f.lineno, f.name, (f.line or "").strip()
    return None                                                                                                                                                                                                           
                
def assertion_text(exc):                                                                                                                                                                                                  
    """Get the underlying AssertionError message, even if wrapped."""
    cur = exc                                                                                                                                                                                                             
    seen = set()
    while cur is not None and id(cur) not in seen:                                                                                                                                                                        
        seen.add(id(cur))
        if isinstance(cur, AssertionError):                                                                                                                                                                               
            return str(cur)
        cur = cur.__cause__ or cur.__context__                                                                                                                                                                            
    return ""
                                                                                                                                                                                                                        
def truncate(s, n=200):                                                                                                                                                                                                   
    s = s.strip().replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."                                                                                                                                                                            
                                                                                                                                                                                                                        
# ---------- kernels ----------------------------------------------------
@triton.jit                                                                                                                                                                                                               
def k_dot_i8(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)                                                                                                                                                      
    a = tl.load(A + om[:, None] * K + ok[None, :])                                                                                                                                                                        
    b = tl.load(B + ok[:, None] * N + on[None, :])                                                                                                                                                                        
    acc = tl.zeros((M, N), dtype=tl.int32)                                                                                                                                                                                
    acc = tl.dot(a, b, acc)                                                                                                                                                                                               
    tl.store(C + om[:, None] * N + on[None, :], acc)                                                                                                                                                                      
                                                                                                                                                                                                                        
@triton.jit                                                                                                                                                                                                               
def k_dot_f16(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)                                                                                                                                                      
    a = tl.load(A + om[:, None] * K + ok[None, :])                                                                                                                                                                        
    b = tl.load(B + ok[:, None] * N + on[None, :])
    c = tl.dot(a, b)                       # fp32 acc                                                                                                                                                                     
    tl.store(C + om[:, None] * N + on[None, :], c.to(tl.float16))                                                                                                                                                         
                                                                                                                                                                                                                        
# ---------- runner -----------------------------------------------------                                                                                                                                                 
def i8(*s):  return torch.randint(-8, 8, s, device='cuda', dtype=torch.int8)
def f16(*s): return torch.randn(*s, device='cuda', dtype=torch.float16)                                                                                                                                                   
def i32e(*s): return torch.empty(s, device='cuda', dtype=torch.int32)
def f16e(*s): return torch.empty(s, device='cuda', dtype=torch.float16)                                                                                                                                                   
                                                                                                                                                                                                                        
results = []   # (dtype, M, N, K, num_warps, status, assert_msg, frame)                                                                                                                                                   
                                                                                                                                                                                                                        
def run_dot(dtype, M, N, K, nw):                                                                                                                                                                                          
    try:        
        if dtype == "int8":
            k_dot_i8[(1,)](i8(M, K), i8(K, N), i32e(M, N),                                                                                                                                                                
                            M, N, K, num_warps=nw, num_stages=2)                                                                                                                                                           
        else:                                                                                                                                                                                                             
            k_dot_f16[(1,)](f16(M, K), f16(K, N), f16e(M, N),                                                                                                                                                             
                            M, N, K, num_warps=nw, num_stages=2)                                                                                                                                                          
        torch.cuda.synchronize()
        results.append((dtype, M, N, K, nw, "OK", "", None))                                                                                                                                                              
    except Exception as e:
        msg = assertion_text(e) or f"{type(e).__name__}: {e}".splitlines()[0]                                                                                                                                             
        frame = first_triton_internal_frame(e)                                                                                                                                                                            
        results.append((dtype, M, N, K, nw, type(e).__name__, msg, frame))                                                                                                                                                
                                                                                                                                                                                                                        
CASES = [                                                                                                                                                                                                                 
    # (M, N, K, num_warps)
    (16,   16,  16, 1),  # K below the documented "K >= 32" threshold                                                                                                                                                     
    (16,   16,  32, 1),                                                                                                                                                                                                   
    (32,   32,  32, 1),                                                                                                                                                                                                   
    (32,   32,  64, 1),                                                                                                                                                                                                   
    (64,   64,  32, 2),                                                                                                                                                                                                   
    (64,   64,  64, 2),                                                                                                                                                                                                   
    (128, 128,  32, 4),
    (128, 128,  64, 4),                                                                                                                                                                                                   
    (128, 128, 128, 4),                                                                                                                                                                                                   
    (128, 256,  64, 4),
    (256, 128,  64, 4),                                                                                                                                                                                                   
]                                                                                                                                                                                                                         

print("\nRunning fp16 baseline (sanity: dot path itself works)...")                                                                                                                                                       
for M, N, K, nw in CASES:
    run_dot("fp16", M, N, K, nw)                                                                                                                                                                                          
                                                                                                                                                                                                                        
print("Running int8 cases...")
for M, N, K, nw in CASES:                                                                                                                                                                                                 
    run_dot("int8", M, N, K, nw)
                                                                                                                                                                                                                        
# ---------- compact table ---------------------------------------------
print("\n" + "=" * 72)                                                                                                                                                                                                    
print(f"{'dtype':<5} {'M':>4} {'N':>4} {'K':>4} {'nw':>3}  {'status':<18}  detail")
print("-" * 72)                                                                                                                                                                                                           
for dtype, M, N, K, nw, status, msg, frame in results:
    detail = ""                                                                                                                                                                                                           
    if status != "OK":
        if msg:                                                                                                                                                                                                           
            detail = msg
        if frame:
            f, ln, fn, src = frame                                                                                                                                                                                        
            detail += f"   @ {f}:{ln} ({fn})"
    print(f"{dtype:<5} {M:>4} {N:>4} {K:>4} {nw:>3}  {status:<18}  {truncate(detail, 140)}")                                                                                                                              
                                                                                                                                                                                                                        
# ---------- per-failure deep dive (one int8 case) ---------------------                                                                                                                                                  
print("\n" + "=" * 72)                                                                                                                                                                                                    
print("Deep-dive on first failing int8 case (full Triton-internal trace):")                                                                                                                                               
print("=" * 72)                                                                                                                                                                                                           
first_fail = next((r for r in results if r[0] == "int8" and r[5] != "OK"), None)
if first_fail is not None:                                                                                                                                                                                                
    dtype, M, N, K, nw, status, msg, frame = first_fail
    print(f"Reproducing:  {dtype}  {M}x{N}x{K}  num_warps={nw}")                                                                                                                                                          
    try:                                                                                                                                                                                                                  
        k_dot_i8[(1,)](i8(M, K), i8(K, N), i32e(M, N),
                        M, N, K, num_warps=nw, num_stages=2)                                                                                                                                                               
    except Exception as e:
        print(f"\nException type     : {type(e).__name__}")                                                                                                                                                               
        print(f"Underlying assert  : {assertion_text(e) or '(empty assertion message)'}")                                                                                                                                 
        print(f"\nFull error message:\n{textwrap.indent(str(e), '  ')}")                                                                                                                                                  
        print("\nTraceback frames inside the Triton package:")                                                                                                                                                            
        triton_root = os.path.dirname(triton.__file__)                                                                                                                                                                    
        for f in traceback.extract_tb(e.__traceback__):                                                                                                                                                                   
            if (f.filename or "").startswith(triton_root):                                                                                                                                                                
                rel = os.path.relpath(f.filename, os.path.dirname(triton_root))
                print(f"  {rel}:{f.lineno}  in {f.name}")                                                                                                                                                                 
                if f.line:                                                                                                                                                                                                
                    print(f"     {f.line}")
                                                                                                                                                                                                                        
# ---------- summary ---------------------------------------------------                                                                                                                                                  
ok_i8  = sum(1 for r in results if r[0] == "int8" and r[5] == "OK")
bad_i8 = sum(1 for r in results if r[0] == "int8" and r[5] != "OK")                                                                                                                                                       
ok_f16 = sum(1 for r in results if r[0] == "fp16" and r[5] == "OK")
bad_f16= sum(1 for r in results if r[0] == "fp16" and r[5] != "OK")                                                                                                                                                       
                
print("\n" + "=" * 72)                                                                                                                                                                                                    
print("SUMMARY")
print("=" * 72)                                                                                                                                                                                                           
print(f"  fp16 dot : {ok_f16} OK / {bad_f16} FAIL  out of {len(CASES)} cases")
print(f"  int8 dot : {ok_i8} OK / {bad_i8} FAIL  out of {len(CASES)} cases")                                                                                                                                              
if ok_f16 == len(CASES) and ok_i8 == 0:                                                                                                                                                                                   
    print("\n  Conclusion: dot path works, but int8 dot is rejected at every tile size")                                                                                                                                  
    print(f"              tested on Triton {triton.__version__} + cc {p.major}.{p.minor}.")                                                                                                                               
    print("              The frontend assertion lives in:")                                                                                                                                                               
    seen_frames = set()                                                                                                                                                                                                   
    for r in results:                                                                                                                                                                                                     
        if r[0] == "int8" and r[7] is not None:                                                                                                                                                                           
            f, ln, fn, src = r[7]
            key = (f, ln)                                                                                                                                                                                                 
            if key not in seen_frames:
                seen_frames.add(key)                                                                                                                                                                                      
                print(f"                {f}:{ln}   ({fn})")
                if src: print(f"                    {src}")                                                                                                                                                               
elif ok_i8 > 0:
    print("\n  Conclusion: int8 dot works for some tile sizes — see table above.")                                                                                                                                        
else:                                                                                                                                                                                                                     
    print("\n  Conclusion: dot path itself is broken in this build (fp16 also failing).")       