# Validation Run — Local Smoke Test

## Windows (this host)

PyTorch failed to load without MSVC Redistributable. Pipeline fell back to **NumPy smoke test** (`tools/numpy_smoke.py`):

- PSD Proposition 1 verified via `eigvalsh`
- Hierarchical tokenization (250 kb region -> n bins)
- Stratified Frobenius loss computed in NumPy

## Full PyTorch pipeline

```bash
./verify_pipeline.sh   # Linux / Docker
```

Or install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) on Windows, then:

```powershell
.\verify_pipeline.ps1
```

## Docker

```bash
docker build -t hgt-psd:v1 .
docker run --rm -v "$(pwd)":/workspace hgt-psd:v1
```
