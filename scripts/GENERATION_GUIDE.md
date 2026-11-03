# Page-package generation guide

You are writing the educational companion for ONE page of *Deep Learning with Python, 3rd edition*
(Chollet & Watson). You receive a generation bundle (`work/context/page_NNN.md`) and the rendered page
image (`work/page_images/page_NNN.png`). You must write `data/pages/page_NNN.json`.

Accuracy beats everything. The reader has the real page open next to your notes. Your notes must be
about **this page's actual content**, with technical depth, never generic deep-learning filler.

## Hard rules

1. **Never attribute to the authors something they did not write.** Every item carries a `source` tag:
   - `book` — directly supported by the page (or a cited page of the book),
   - `ai_explanation` — extra technical explanation you add,
   - `analogy` — illustrative analogy/example you invent,
   - `inference` — your interpretation of what the authors imply.
   Paragraph notes and summaries are about the book text; when you add something, say so in the prose
   ("Beyond the book: …", "Note (not in the book): …").
2. **Cite PDF page numbers** (`p.61`) when you refer elsewhere in the book — use PDF numbering (the bundle
   lists both; the UI shows printed numbers too). Only cite pages the bundle shows you or that you can see
   in the chapter outline; never invent a page number.
3. **Reproduce original code verbatim** in `code[].original_code` (copy from the `code` elements; fix
   only obvious wrapped-line breaks). Any code you write yourself goes in `illustrative_code`, never mixed in.
4. **Formulas**: `latex` must be correct LaTeX for KaTeX (no `\begin{align}`; use `\\` line breaks inside
   `\begin{aligned}` if needed). `source_text` is how the formula appears in the book. Look at the page image:
   formulas in this book are written inline, like `output = relu(dot(input, W) + b)`; reconstruct them properly
   (e.g. `\text{output} = \operatorname{relu}(W\cdot x + b)`). Define every variable. Add a small
   numerical example when it helps. If a page has no formula, `math` is `[]`. Do not manufacture equations.
5. **Visualizations are only for concepts actually on the page.** No decorative diagrams. If nothing on
   the page benefits from a visual, `visualizations` is `[]`. Prefer an interactive `widget` from the catalog
   below with parameters matching the page (e.g. the exact layer sizes, the exact matrix in the text);
   use `mermaid` for process/architecture flows; `chart` (Chart.js config) for curves/tables of numbers;
   `svg` (inline, ≤ 6 KB, no scripts, viewBox-based, uses `currentColor`) for a custom static diagram.
6. **Paragraph notes cover every substantial paragraph** (body, bullet, sidebar, quote elements with more
   than ~25 words), in page order, each referencing the element id (`ref`) and quoting its first ~12 words
   verbatim in `excerpt`. Short transitional lines may be merged into the neighbouring note.
7. **Questions**: at least 5 and up to 8: mix of `short`, `conceptual`, `mcq` (4 options, `answer` equal to
   one option verbatim), `math` when the page has math, `self` (self-assessment). Each has an `explanation`
   that says why the answer is right (and, for mcq, why the distractors are wrong) and `page_refs`.
8. **Light pages**: for `front_matter`, `index`, `blank`, `figure_only` and `chapter_opener` pages, set
   `generation.status = "light"`, still write an honest overview + summary (what the page is and how to
   use it), keep the other arrays short or empty. A chapter opener should summarize what the chapter covers
   (the bullet list on the page) and set `visualizations` to a mermaid chapter map if useful.
9. **Uncertainty**: if the extraction is noisy or you are not sure (e.g. a figure you cannot read), add a
   message to `generation.review_flags` and lower `generation.confidence`. Never guess silently.
10. Write in clear, precise English. Markdown is allowed in prose fields (`summary`, `what_it_means`,
    `body`, `explanation` …): paragraphs, lists, `inline code`, **bold**, and `$...$` / `$$...$$` math.
11. Length: a full content page typically yields 2 500–6 000 words of JSON prose. Depth over padding.

`overview.subsection` may be `""` when the page is not inside a numbered subsection. The `code` array must have an entry for **every** code element on the page (numbered listing or not).

**Connections format:** top-level `connections.earlier/later/prerequisites/related_*` entries are objects `{"page": 57, "note": "…"}`; `paragraphs[].connections` entries are plain strings like `"p.57: …"`.

## JSON schema (all keys required unless marked optional; see scripts/common.py for the Pydantic model)

```
{
 "page_id": "page_061", "pdf_page": 61, "printed_page": "37", "book_id": "dlwp3",
 "generation": {"status": "complete"|"light", "model": "<your model name>", "generated_at": "<ISO-8601>",
                "confidence": "high"|"medium"|"low", "review_flags": [], "extraction_quality": "short note"},
 "overview": {"title": "<short page title>", "chapter": "...", "section": "...", "subsection": "...",
              "main_topic": "...", "central_idea": "1-2 sentences", "importance": "why this page matters",
              "learning_objectives": ["...", "..."], "prerequisites": ["...", "..."],
              "page_kind": "content"|"chapter_opener"|"front_matter"|"index"|"blank"|"figure_only"},
 "summary": "markdown, 150-450 words for content pages, faithful and specific",
 "paragraphs": [{"ref": "p061_e03", "excerpt": "first ~12 words verbatim", "what_it_says": "...",
                 "what_it_means": "...", "why_it_matters": "...", "technical_idea": "optional",
                 "terminology": [{"term": "...", "meaning": "..."}], "connections": ["p.57: tensors were defined as …"]}],
 "concepts": [{"id": "affine_transform", "name": "Affine transform", "definition": "...", "intuition": "...",
               "technical": "...", "significance": "...", "example": "...", "misconception": "...",
               "related": ["linear_transform", "dense_layer"], "source": "book"}],
 "math": [{"id": "m1", "title": "...", "latex": "y = Wx + b", "source_text": "y = W @ x + b",
           "variables": [{"symbol": "W", "meaning": "..."}], "operations": "...", "purpose": "...",
           "interpretation": "...", "intuition": "...", "numerical_example": "optional markdown",
           "source": "book", "book_ref": "p.61"}],
 "code": [{"id": "c1", "listing": "3.23" (or null), "title": "...", "original_code": "verbatim",
           "language": "python", "what_it_does": "...",
           "line_explanations": [{"lines": "1-2", "explanation": "..."}],
           "inputs_outputs": "...", "algorithm": "...", "theory_link": "...",
           "parameters": [{"name": "epochs", "role": "..."}], "pitfalls": ["..."],
           "illustrative_code": "optional, AI-written"}],
 "deeper": [{"title": "...", "kind": "analogy"|"step_by_step"|"application"|"numerical"|"breakdown"|"alternative",
             "body": "markdown", "source": "analogy"|"ai_explanation"|"inference"}],
 "visualizations": [{"id": "v1", "type": "widget", "title": "...", "why": "what on this page it shows",
                     "widget": "<catalog name>", "params": {...}},
                    {"id": "v2", "type": "mermaid", "title": "...", "why": "...", "mermaid": "flowchart LR\n A-->B"},
                    {"id": "v3", "type": "chart", "title": "...", "why": "...", "chart": {"type": "line", "data": {...}, "options": {...}}},
                    {"id": "v4", "type": "svg", "title": "...", "why": "...", "svg": "<svg viewBox=...>…</svg>"}],
 "misconceptions": [{"claim": "...", "correction": "..."}],
 "connections": {"earlier": [{"page": 57, "note": "..."}], "later": [{"page": 90, "note": "..."}],
                 "prerequisites": [...], "related_math": [...], "related_code": [...], "related_figures": [...]},
 "takeaways": ["...", "..."],
 "questions": [{"id": "q1", "type": "mcq", "question": "...", "options": ["a", "b", "c", "d"], "answer": "b",
                "explanation": "...", "page_refs": [61]},
               {"id": "q2", "type": "short", "question": "...", "options": [], "answer": "...", "explanation": "...", "page_refs": [61]}]
}
```

Concept ids are lowercase snake_case and should be **reused across pages** for the same concept
(e.g. `gradient_descent`, `backpropagation`, `tensor`, `dense_layer`, `relu`, `softmax`, `loss_function`,
`overfitting`, `convolution`, `attention`, `transformer`, `embedding`, `rnn`, `lstm`, `batch_normalization`,
`residual_connection`, `data_augmentation`, `transfer_learning`, `learning_rate`, `optimizer`, `sgd`, `adam`,
`cross_entropy`, `mse`, `validation_set`, `k_fold_validation`, `regularization`, `dropout`, `keras_layer`,
`functional_api`, `sequential_model`, `keras_model`, `autodiff`, `gradient_tape`, `tokenization`,
`language_model`, `self_attention`, `positional_encoding`, `diffusion_model`, `vae`, `gan`).

## Widget catalog (type "widget"; `params` keys shown with defaults)

| widget | params | shows |
|---|---|---|
| `gradient_descent` | `surface`: "bowl"\|"valley"\|"ravine"\|"saddle"\|"twohills"; `lr` 0.1; `start` [x,y] (range -3..3); `steps` 25; `compare_lrs` [0.02,0.1,0.5] | 2-D contour map, path of GD, sliders for learning rate and start, comparison of learning rates, loss-vs-step curve |
| `gradient_descent_1d` | `expr` "x*x" (JS expression in x); `lr` 0.1; `start` 2.5; `steps` 20; `momentum` 0 | 1-D curve with steps and tangent lines |
| `nn_architecture` | `layers`: [{`name`, `type`: "input"\|"dense"\|"conv"\|"pool"\|"flatten"\|"dropout"\|"embedding"\|"rnn"\|"output"\|"other", `units` (omit for input; give `shape` instead), `activation`, `shape`: "(None, 784)"}] | clickable layer boxes: output shape, dense parameter count (prev×units+units), activation name, data flow |
| `backprop_graph` | `example`: "linear_mse"\|"two_layer_relu"; `x` 1.5; `y` 1.0; `w` 0.8; `b` 0.1; `lr` 0.1 | computational graph with forward/backward stepper, chain-rule gradients, parameter update |
| `activation_functions` | `functions` ["relu","sigmoid","tanh"] (also "softplus","gelu","leaky_relu","elu","linear","swish"); `show_derivative` true | plotted functions and derivatives, hover values |
| `loss_functions` | `losses` ["mse","mae"] (also "bce","huber","cce"); `y_true` 1 | loss as function of prediction |
| `softmax_temperature` | `logits` [2,1,0.1]; `labels` optional; `temperature` 1 | logit sliders → probabilities, temperature |
| `convolution` | `input`: 2-D array (≤ 8×8) or `preset`: "edge"\|"diag"\|"cross"; `kernel` [[...]] (3×3); `stride` 1; `padding` "valid"\|"same" | step through the sliding window, see each output value computed |
| `pooling` | `input` 2-D (≤ 8×8) or `preset`; `size` 2; `stride` 2; `mode` "max"\|"avg" | pooling windows and outputs |
| `tensor_shapes` | `steps`: [{`op`: "input", `shape`: [60000,28,28]}, {`op`: "reshape", `shape`: [60000,784]}, …] (`op` is free text, keep it short; use `null` for a batch axis of unknown size) | shape-flow diagram, element counts |
| `attention` | `tokens` ["the","cat","sat","on","mat"]; `dim` 4; `seed` 1 | click a query token, see scores → softmax weights → weighted sum; heatmap |
| `embedding_space` | `points` [{`label`, `x`, `y`, `group`}]; `arrows` [[labelA,labelB]] | 2-D embedding scatter with vector arithmetic arrows |
| `rnn_unroll` | `timesteps` 4; `cell` "simple"\|"lstm"\|"gru" | unrolled recurrence, click a step to see the state update equation |
| `training_curves` | `epochs` 20; `overfit_from` 6; `noise` 0.03 | train/val loss & accuracy with overfitting point, slider to move the best-epoch marker |
| `overfit_underfit` | `degrees` [1,3,12]; `n_points` 20; `noise` 0.3; `seed` 1 | polynomial fits of varying capacity, train/test error |
| `kfold` | `k` 4; `n` 16 | fold layout, which samples validate in each fold |
| `data_split` | `train` 0.7; `val` 0.15; `test` 0.15; `n` 1000 | proportions and roles of the three sets |
| `matrix_product` | `A` [[1,2],[3,4]]; `B` [[5,6],[7,8]] (≤ 4×4) | animated dot products producing each output cell |
| `broadcasting` | `a_shape` [32,10]; `b_shape` [10] | how the smaller tensor is virtually repeated |
| `derivative_tangent` | `expr` "x*x"; `x0` 1.5; `h` 0.5 | secant → tangent, derivative as local slope |
| `vector_geometry` | `matrix` [[0,-1],[1,0]]; `translate` [0,0]; `relu` false; `points` optional (polygon vertices, default an L-shape) | 2-D transform of a shape; the UI has sliders for all four matrix entries, both translation components and a relu checkbox |
| `confusion_matrix` | `matrix` [[tp,fn],[fp,tn]]; `labels` ["positive","negative"] | precision/recall/accuracy computed live |
| `residual_block` | `layers` 2 | block diagram with skip connection toggle |
| `transformer_block` | `kind` "encoder"\|"decoder"\|"encoder_decoder" | hoverable block diagram of the Transformer |
| `dropout` | `units` 10; `rate` 0.5 | re-sample masks, see scaling |
| `learning_rate_schedule` | `schedule` "step"\|"cosine"\|"warmup_cosine"\|"exponential"; `epochs` 50; `lr` 0.1 | lr-vs-epoch plot with parameters |
| `token_sequence` | `text` "The cat sat"; `mode` "word"\|"char"\|"subword"; `vocab` optional | tokens → ids → one-hot / embedding lookup |
| `noise_schedule` | `steps` 50; `schedule` "linear"\|"cosine" | diffusion forward process, signal/noise ratio |
| `image_tensor` | `height` 4; `width` 4; `channels` 3 | (H, W, C) tensor with channel slices |

Give each widget a `title` and a `why` that ties it to the page ("Figure 2.10 rotates a vector by θ; try
other angles here"). You may include several visuals when the page truly has several visual ideas.
