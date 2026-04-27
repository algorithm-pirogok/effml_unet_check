# UNet Segmentation Inference Benchmarks

Benchmarks the pretrained Carvana UNet model from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) on a segmentation task.

Compares:
- **Baseline fp16 inference** (when CUDA is available)
- **`torch.compile`-accelerated inference** (when supported by the installed PyTorch)

## Project structure

```
src/
  model.py        — pretrained UNet loading
  dataset.py      — CarvanaTrainDataset and DataLoader builder
  benchmark.py    — inference benchmark runner with Dice/F1 metric
  experiment.py   — experiment orchestration (configs, history)
run_benchmark.py  — main entry point: runs all experiments and saves results
plot_results.py   — plots avg time per image vs batch size
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_benchmark.py
python plot_results.py
```

The benchmark script expects Carvana dataset images and masks. Update `train_root` and `mask_root` paths in `run_benchmark.py` accordingly.

---

# UNet INT8 quantisation — что получилось

Эксперимент: ускорить инференс Carvana UNet на Colab T4 через INT8-квантизацию,
сравнить с fp16-бейзлайном, разложить ускорение на компоненты.

Главный артефакт: [`unet_segmentation_quantised.ipynb`](unet_segmentation_quantised.ipynb)
— self-contained Colab-ноутбук со всеми прогонами.

## Стенд

- **GPU:** Tesla T4 (sm_75) — Colab free-tier.
- **TensorRT:** 10.16.1, **PyTorch:** 2.10.0 + CUDA 12.8.
- **Модель:** Carvana UNet (n_channels=3, n_classes=2), обучен в этом же
  ноутбуке от `nn.init` за 3 эпохи AdamW (lr=1e-4, CE + soft-Dice). Вход
  640×959, fp32-веса в `last.pth` (≈ 124 MB).
- **Метрики:** скорость — `time.perf_counter` ± `torch.cuda.synchronize()`;
  качество — `BinaryF1Score` (= foreground Dice) на val-сплите 400 картинок.

## Подход к квантизации

Тип: **Post-Training Quantization (PTQ)**, статическая, симметричная,
по-тензорная. Без QAT, без правок модели.

Пайплайн (cell 36 в ноутбуке):

1. **ONNX-экспорт** через **legacy** `torch.onnx.export(..., dynamo=False, opset_version=17)`. Принципиально legacy: dynamo-путь в torch≥2.6 пишет веса наружу как external data + opset 18+, и TRT-парсер падает с пустыми инициализаторами. Размер ONNX совпадает с fp32-весами (≈124 MB) — sanity-чек, что веса встроены.
2. **Калибровка** через `IInt8EntropyCalibrator2`: 32 батча × 2 = 64 изображения из val-сплита. TRT собирает гистограммы активаций и подбирает per-tensor scale-факторы по минимуму KL-дивергенции с исходным fp32-распределением. Калибровочный буфер — один pre-allocated CUDA-tensor, TRT читает `data_ptr()`.
3. **Build** с флагами `INT8 | FP16` (без fp32): квантизуемые свёртки/линейные слои → INT8, неквантизуемые (например, финальный 1×1 на 2 класса, если попадёт под фильтр TRT) → FP16. Workspace 1 GiB (T4-safe). Optimization profile `min/opt/max = (1, 2, 4)`. Build идёт ~553 с (калибровка + autotune ядер по форме).
4. **Engine** — 31.4 MB (≈ ¼ от ONNX, веса теперь int8). Калибровочный кэш в отдельном `.calib` — переиспользуется на повторных запусках.
5. **Runtime** — `execute_async_v3` напрямую с torch CUDA-тензорами (USER_MANAGED-аллокатор: TRT и torch не воюют за HBM).

## Скорость на разных батчах

Все цифры — `total_inference / images`, тот же val-сплит, тот же кернел
(cell 43, все три пути друг за другом).

| batch | torch-fp16 ms/img | TRT-fp16 ms/img | TRT-int8 ms/img | TRT-int8 img/s |
|------:|-----------------:|----------------:|----------------:|---------------:|
| 2     | 88.076 | 38.731 | **19.660** | **50.9** |
| 4     | 89.673 | 41.528 | 19.767 | 50.6 |
| 8     | 91.051 | 44.454 | 22.620 | 44.2 |

**Лучший per-image — batch=2: 19.66 мс/img, 50.9 img/s.** Причины:

- batch=2 укладывается в `MAX_BATCH=4` одним вызовом — нет chunk-overhead'а
  и `torch.cat`-копий между сегментами.
- TRT-engine собирался с `opt=(2, …)` — autotune ядер делался под эту
  форму, отсюда лучшая утилизация Tensor Cores.
- batch=8 режется на 2 × batch=4 (MAX_BATCH=4 — лимит для T4 по HBM),
  отсюда ещё ~10% сверху на каждый image.

torch-fp16 практически flat по батчу (88 ↔ 91 мс/img) — он compute-bound,
батч ему не помогает. Поэтому весь выигрыш INT8 виден как чистый speedup.

## Декомпозиция: что от компилятора, что от INT8

Чтобы отделить вклад **TRT-fusion+NHWC+autotune** от **самой INT8-квантизации**,
собрали третий путь — TRT-engine с **одним только** `BuilderFlag.FP16` (без
INT8, без калибратора, тот же ONNX, тот же profile, тот же workspace).
Отличие от int8-engine ровно одно: тип ядра (HMMA fp16 vs IMMA int8).

| batch | fusion × (torch-fp16 → TRT-fp16) | IMMA × (TRT-fp16 → TRT-int8) | total × |
|------:|--------------------------------:|-----------------------------:|--------:|
| 2     | **2.27×** | **1.97×** | **4.48×** |
| 4     | 2.16× | 2.10× | 4.54× |
| 8     | 2.05× | 1.97× | 4.03× |

**Что показала декомпозиция:**

- **IMMA × ≈ 2.0** на всех батчах — попадает ровно в теоретический пик
  INT8 vs FP16 Tensor Cores на Turing (sm_75: 130 TOPs INT8 / 65 TFLOPs FP16).
  Это значит, что на Conv-героях UNet'а мы упёрлись именно в матумножение,
  а не в memory-bound операции — иначе IMMA не дал бы своих 2×.
- **Fusion × ≈ 2.0–2.3** — чистый эффект TRT-компилятора, не квантизации.
  В PyTorch каждый `Conv → BN → ReLU` — три отдельных kernel-launch'а с
  тремя round-trip'ами активации в HBM. TRT склеивает их в один kernel, BN
  поглощается в веса, активация остаётся в SMEM между Conv и ReLU. Плюс TRT
  автоматически выбирает NHWC layout (PyTorch остаётся в NCHW).
- **Эффекты независимы и мультиплицируются:** 2.27 × 1.97 = 4.47 ≈ 4.48
  (наблюдаемый total на batch=2). Сходимость до 1% — значит модель эффектов
  правильная.

## Качество не страдает

Dice до 4-го знака стабилен на всех трёх путях:

| batch | torch-fp16 | TRT-fp16 | TRT-int8 |
|------:|-----------:|---------:|---------:|
| 2 | 0.9911 | 0.9911 | 0.9910 |
| 4 | 0.9911 | 0.9911 | 0.9910 |
| 8 | 0.9911 | 0.9911 | 0.9910 |

Δ Dice (TRT-int8 − torch-fp16) = **−0.0001** — уровень шума калибровки на
64 изображениях. Для бинарной сегментации автомобилей этого хватает с
огромным запасом.

> **Замечание про QAT.** Изначально мы держали в голове Quantisation-Aware
> Training как fallback, если PTQ просядет по качеству. Поэтому отдельно
> сохранили обучающий пайплайн в [`training/`](training/) — обучение от
> нуля с AdamW + CE+Dice + AMP. На практике QAT не понадобился: PTQ через
> TensorRT вообще не уронил Dice. Папка остаётся для воспроизводимости
> чекпоинта и на случай, если в будущем модель/датасет станет более
> чувствительной к INT8 (например, более тонкая сегментация классов с
> малой долей пикселей).

## Практические выводы

1. На Colab T4 **PTQ через TensorRT даёт 4.0–4.5× к PyTorch-fp16** при
   потере качества **−0.0001 Dice**. Никаких изменений в модели,
   никакого QAT, никакой ручной разметки слоёв.
2. **Половина выигрыша приходит от TRT-компилятора** (fusion + NHWC +
   autotune), а не от самой INT8-квантизации. Это значит, что даже без
   INT8 — просто сборка TRT-fp16-engine из ONNX уже даёт ~2×.
3. **Узкое место fp16 на T4 — compute, а не memory.** Батч его не
   ускоряет, поэтому INT8-IMMA — единственный реальный путь дальше на
   этом железе.
4. **Лучший per-image режим — batch=2** (совпадает с `opt`-формой TRT-профиля,
   укладывается в `MAX_BATCH=4` одним вызовом). На L4/A100 имеет смысл
   увеличить `MAX_BATCH` и `opt` — там HBM позволит.

## Где смотреть

- **Ноутбук:** [`unet_segmentation_quantised.ipynb`](unet_segmentation_quantised.ipynb)
  - cells 0–22 — обучение / загрузка чекпоинта.
  - **cell 36** — TRT-INT8 пайплайн (ONNX → калибровка → build → timed pass на batch=8 + fp16-baseline в той же ячейке).
  - **cells 38–41** — повторы на batch=4 и batch=2.
  - **cells 42–43** — TRT-fp16 vs TRT-int8 vs PyTorch-fp16 декомпозиция.
- **Обучение от нуля:** [`training/`](training/) — `train.py`, `validate.py`,
  `unet.py`, `dataset.py`. Stand-alone, не зависит от `src/` (там
  бенчмарк-пайплайн под предобученные веса с `torch.hub`).

---

# Triton `tl.dot` int8 на Colab T4 — что получилось

Побочное исследование к основной задаче квантизации. Проверяли, можно ли
для int8-матумножения использовать Triton-ядро с `tl.dot(int8, int8) → int32`
вместо TensorRT. Краткий ответ: **на Colab T4 не работает**, и причина — не
в подборе тайла, а в frontend-проверке самого Triton.

Скрипт: [`trton_exp_of_dor_int8.py`](trton_exp_of_dor_int8.py).
Эталонный вывод с Colab T4: [`trton_exp_of_dor_int8_colab_output.txt`](trton_exp_of_dor_int8_colab_output.txt).

## Стенд

```
triton  : 3.6.0
torch   : 2.10.0+cu128
GPU     : Tesla T4  cc=7.5
smem    : 65536 B/block (opt-in)
```

## Что проверяли

Один и тот же диагностический скрипт прогоняет 11 тайлов
`(M, N, K)` — от `16×16×16` до `256×128×64` — двумя путями:

1. **fp16-baseline:** `tl.dot(a_f16, b_f16) → f32` — sanity-чек, что сам
   `tl.dot` собирается на этой карте.
2. **int8-кандидат:** `tl.dot(a_i8, b_i8, acc_i32) → i32` — то, что нам нужно.

По документации Triton требование одно: `K ≥ 32`.

## Что вышло

| Путь    | OK     | FAIL | Где падает |
|---------|-------:|-----:|------------|
| fp16    | 11/11  | 0    | —          |
| int8    | 0/11   | 11   | frontend (`ast_to_ttir`) |

**Все 11 int8-тайлов отклоняются**, включая `K = 32, 64, 128` — то есть
требование `K ≥ 32` соблюдено, но ассерт всё равно срабатывает.
Сообщение `"Input shapes should have M >= 1, N >= 1 and K >= 32"`
**вводит в заблуждение**: внутренняя проверка строже, чем формулировка
ассерта. Падение — не на `make_assembly`, не на `nvcc`, а на самом раннем
этапе фронтенда:

```
triton/compiler/compiler.py:80  in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, ...)
```

То есть Triton 3.6.0 на Turing для int8-`tl.dot` даже не доходит до
выбора `BLOCK_K` / `num_stages` — отбрасывает программу на стадии
построения TTIR. На fp16 тот же путь компилируется на всех 11 тайлах.

## Почему это важно

В `triton-lang/triton#2364` (сентябрь 2023) поддержка int8 `tl.dot` для
Turing была явно добавлена. Наш результат на Triton 3.6.0 — **регрессия**
относительно этого PR: поддержки де-факто нет, при этом ассерт врёт о
том, что именно нужно изменить.

## Практический вывод

На Colab T4 **писать INT8-матумножение на Triton не вариант**. Реальные
рабочие пути для int8 на T4:

- **TensorRT INT8** — что и сделано в [`unet_segmentation_quantised.ipynb`](unet_segmentation_quantised.ipynb). ONNX → калибровка → IMMA-ядра. См. секцию «UNet INT8 quantisation» выше.
- **Triton fp16 `tl.dot`** — если нужен именно Triton-кернел: fp16 на T4
  работает без проблем; int8-значения умещаются в fp16 точно (`|x| ≤ 127`,
  fp16 mantissa 11 бит — точное представление), результат побитово совпадает
  с честным int8-вариантом.
- **Сменить рантайм на L4 / A100 (Colab Pro)** — там int8 `tl.dot`
  компилируется штатно (Ampere/Ada `mma.m16n8k32.s8.s8.s32`).

## Источники

- [Triton API — `triton.language.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
  — официальная сигнатура и допустимые dtype (int8 для входов, int32 для аккумулятора).
- [Triton PR #2364 — Implement dot for int8 on Turing](https://github.com/triton-lang/triton/pull/2364)
  — PR, который изначально включил int8 на T4 (2023-09); по нашим тестам
  в Triton 3.6.0 эта поддержка не работает.

---

Bottom line: on Colab T4, **TensorRT INT8 PTQ gives 4.0–4.5× over PyTorch-fp16
without measurable quality loss**, and the path doesn't depend on Triton.
