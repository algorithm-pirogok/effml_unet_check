# Sparsification block — defense slides outline

7 slides for the defense (27.04.2026, 19:40). Scope: my block only — Anton
covers baseline+quantization, Andrey covers compilers. Each slide gives the
exact text for the slide body, the figure to drop in, and a one-sentence
speaker note (что говорить).

---

## Slide 1 — Title

**On slide:**

> **Спарсификация UNet**
> magnitude / 2:4 / structured + fine-tune
>
> Курилов Олег
> Курс «Эффективные модели ML»

**Speaker note (15 сек):**
> «Я отвечал за блок спарсификации — три метода, fine-tune,
> и проверка реального speedup через cuSPARSELt. Покажу что работает,
> что не работает, и почему».

---

## Slide 2 — Три метода и главные вопросы

**On slide (две колонки):**

Левая колонка — **«Что мерим»**:

| Метод | Что делает |
|---|---|
| Magnitude (L1) | Зануляет N% весов с самой маленькой `\|w\|` |
| 2:4 semi-structured | В каждых 4 элементах вдоль in-channel — 2 нуля |
| Structured channel | Зануляет целые out-каналы свёрток |

Правая колонка — **«На что смотрим»**:
- Качество (Dice / BinaryF1) — сколько падает?
- Скорость (ms/img) — реальный speedup?
- Память (peak GB) — экономим ли?
- Восстановимо ли через fine-tune?

**Speaker note:**
> «Три классических метода. Хочу понять для каждого: на сколько просядет
> Dice, можно ли вернуть retrain'ом, и появится ли вообще скорость на нашей GPU».

---

## Slide 3 — Качественный envelope (figure 01)

**На слайде:** `docs/slides/01_quality_envelope.png`

**Подпись под графиком:**
> **UNet выдерживает до 70% magnitude sparsity почти бесплатно.**
> Cliff на 90% — при таком прунинге без retrain модель ломается.

**Speaker note (30 сек):**
> «Магнитудное прунинг: график Dice от sparsity. До 50% — Dice не падает.
> До 70% — теряем меньше 1%. На 90% обрыв — модель почти умирает.
> Это говорит, что 70% весов UNet'а избыточны».

---

## Slide 4 — Fine-tune возвращает качество (figure 02)

**На слайде:** `docs/slides/02_finetune_recovery.png`

**Подпись:**
> **1 эпоха fine-tune (frozen encoder) → все 3 метода ВЫШЕ baseline.**
> magnitude@0.9: 0.235 → 0.994 за один проход.

**Speaker note (40 сек):**
> «Если pruning ломает Dice — не страшно. NVIDIA ASP рецепт: после прунинга
> делаем 1 эпоху retrain с замороженной маской. PyTorch держит маску в
> PruningContainer'е, нули не возвращаются. Самый драматичный случай —
> magnitude 90%: модель почти сломана с Dice 0.235, через эпоху Dice 0.994 —
> выше baseline».

---

## Slide 5 — Iterative pruning beats one-shot (figure 03 + ASP combo figure 04)

**На слайде** — два графика рядом, либо последовательно:

Левый: `docs/slides/03_iterative_vs_oneshot.png`
> **Gradual ladder лучше one-shot:**
> 30% → 50% → 70% даёт Dice 0.9950, one-shot 70% — 0.9940.

Правый: `docs/slides/04_asp_combo.png`
> **Полный NVIDIA ASP combo работает:**
> baseline 0.9912 → magnitude 50% + ft → +2:4 + ft → 0.9944.

**Speaker note (40 сек):**
> «Дополнили двумя экспериментами. Слева — iterative: вместо одного шага
> прунинга делаем три, между шагами короткий retrain. Финальный Dice 0.9950
> — небольшой, но воспроизводимый выигрыш. Справа — combo magnitude+2:4+ft —
> это и есть полный NVIDIA рецепт. End-to-end Dice растёт выше baseline».

---

## Slide 6 — Speedup и Profiler

**На слайде:**
- Слева: скриншот Perfetto trace (твой `profile_baseline_fp16.trace.json`)
- Справа: `docs/slides/05_cusparselt_speedup.png`

**Главная мысль:**
> **Sparsity не даёт wall-time speedup на нашей GPU.**

**Подпись слева:**
> 95% времени UNet проводит в `aten::conv2d` (cudnn dense kernels).
> Этот трейс выглядит ИДЕНТИЧНО для baseline / magnitude / 2:4 — kernels
> не сокращаются от наличия нулей.

**Подпись справа:**
> cuSPARSELt sparse 2:4 на A4000: **везде медленнее** чем dense, даже на
> LLM-shape матрицах. Hardware 2:4 path фактически работает только на
> datacenter Ampere (A100), не на pro/consumer (sm_86).

**Speaker note (50 сек):**
> «Самый интересный результат блока. Профайлер показывает что 95% времени
> UNet — это conv2d через cudnn. И этот таймлайн идентичен для всех трёх
> sparsified вариантов — dense kernels не используют нули. Чтобы получить
> ускорение, нужны sparse kernels. Я попробовал NVIDIA cuSPARSELt — на
> A4000 он стабильно медленнее dense на всех shape, включая LLM-размер.
> Вывод: реальный 2:4 deploy требует A100, не нашу A4000».

---

## Slide 7 — Method landscape + выводы (figure 06)

**На слайде:** `docs/slides/06_method_landscape.png`

**Bullet выводы (правая часть слайда):**

- **Качество**: 70% magnitude — практически бесплатно, после retrain — выше baseline
- **Скорость**: на dense kernels — нулевая. На consumer Ampere через cuSPARSELt — даже отрицательная
- **Память**: peak GPU memory одинаковый (124 MB params, 6.1 GB peak — везде)
- **Что сэкономим**: только размер чекпоинта на диске (если хранить в sparse-формате)

**Что бы делал дальше:**
- Physical channel removal (`torch_pruning`) — единственный путь к реальному speedup на dense kernels
- Перенос на A100 → cuSPARSELt должен заработать
- Combine с квантизацией (Антона) → 2 ортогональных способа сжатия

**Speaker note (30 сек):**
> «Один график — все эксперименты. Зелёные звёзды — после fine-tune все
> методы у baseline или выше. Красные точки внизу — без retrain ломается.
> Главный takeaway: спарсификация в стоковых kernels — это про размер
> чекпоинта, не про скорость. Реальный runtime speedup требует либо A100
> для 2:4, либо физического удаления каналов для structured. Это два пути
> для следующей итерации проекта».

---

## Tips для презентации

- **Темп**: 7 слайдов × ~30-45 сек = 4-5 минут на блок. Норм для общей 15-минутной презентации команды.
- **Главный wow-эффект**: slide 4 (fine-tune recovery), 0.235 → 0.994. Не торопись.
- **Защитный аргумент** на slide 6: «Negative result тоже результат — теперь команда знает что 2:4 на A4000 не работает, для прода нужна A100».
- **Вопросы в Q&A которые скорее всего зададут**:
  - "Почему не сделал physical removal?" → time budget + UNet skip connections нетривиальны
  - "Sparsity 27% при target 70%, почему?" → frozen encoder, prune только decoder, decoder ≈ 40% от total conv → 70% × 40% = ~28%
  - "Совместимо ли с torch.compile/TVM Andrey'я?" → да, magnitude сохраняет shapes, можно компонировать
  - "Recovery выше baseline — это переобучение под val?" → fine-tune был на train split, val видел только в evaluate. Iterative показывает progressive improvement не train→val leak.

## Repo / artefacts to mention

- PR: github.com/algorithm-pirogok/effml_unet_check/pull/1
- Полный отчёт: `docs/sparsity.md` (200+ строк)
- 9 JSON'ов в `results/` (full benchmark, finetune, iterative, asp, cusparselt, memory, profile, smoke)
- 6 фигур + 3 chrome traces
