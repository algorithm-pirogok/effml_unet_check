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

- **TensorRT INT8** — что и сделано в [`unet_segmentation_quantised.ipynb`](unet_segmentation_quantised.ipynb). ONNX → калибровка → IMMA-ядра. См. [`QUANTISATION_NOTES.md`](QUANTISATION_NOTES.md).
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
